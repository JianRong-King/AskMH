"""Generate eval/qa_set.json: paraphrased questions labelled with their source article.

The question is deliberately NOT the article title - a verbatim title would sit almost
on top of the chunk text and inflate Recall@k. Review the output before trusting it.
"""

import json
import random
from collections import defaultdict

from config import ARTICLES_DIR
from eval.harness import QA_SET
from index.chunker import load_article

TARGET = 50
MIN_BODY_CHARS = 150
SEED = 42

PROMPT = """You write realistic questions that a Malaysia Airlines passenger would type into a help chat.

Write ONE question that this FAQ article answers.

Rules:
- Do not reuse the article's title wording. Paraphrase it the way a real passenger would ask.
- Keep it under 20 words, plain English, no airline jargon unless a passenger would say it.
- Ask about one specific thing the article covers.
- Output only the question, nothing else.

Article title: {title}
Article body: {body}"""


def pick_articles() -> list:
    by_topic = defaultdict(list)
    for path in sorted(ARTICLES_DIR.rglob("*.md")):
        meta, body = load_article(path)
        if len(body) >= MIN_BODY_CHARS:
            by_topic[meta.get("topic") or "Uncategorized"].append((meta, body))

    rng = random.Random(SEED)
    for items in by_topic.values():
        rng.shuffle(items)

    # Round-robin across topics so no single topic dominates the set.
    picked, topics = [], sorted(by_topic)
    while len(picked) < TARGET:
        added = False
        for t in topics:
            if by_topic[t]:
                picked.append(by_topic[t].pop())
                added = True
                if len(picked) == TARGET:
                    break
        if not added:
            break
    return picked


def main():
    from eval.harness import timed_call
    from agent.nodes import get_llm

    llm = get_llm()
    rows = []
    for i, (meta, body) in enumerate(pick_articles(), 1):
        prompt = PROMPT.format(title=meta["title"], body=body[:1200])
        resp, m = timed_call(llm.invoke, prompt)
        if resp is None:
            print(f"{i:>3}. SKIP ({m.error})")
            continue
        question = resp.content.strip().strip('"').splitlines()[0].strip()
        rows.append(
            {
                "id": f"q{i:03d}",
                "question": question,
                "expected_url": meta["url"],
                "title": meta["title"],
                "topic": meta.get("topic") or "Uncategorized",
            }
        )
        print(f"{i:>3}. [{m.wall_s:>5.2f}s] {question}")

    QA_SET.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(rows)} questions to {QA_SET}")


if __name__ == "__main__":
    main()
