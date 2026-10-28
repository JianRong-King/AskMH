"""Generate eval/qa_set.json: paraphrased questions labelled with their source article.

The question is deliberately NOT the article title - a verbatim title would sit almost
on top of the chunk text and inflate Recall@k. Review the output before trusting it.

    python -m eval.build_qa_set              # the 50-question set
    python -m eval.build_qa_set --holdout    # eval/qa_holdout.json, 30 short casual questions

The held-out set exists because index/doc2query.py adds LLM-written questions to the index,
which look like the ones here and would flatter recall. It uses unseen articles, a different
seed and a terser, chat-style prompt, so it stays a fair check on that change.
"""

import argparse
import json
import random
from collections import defaultdict

from config import ARTICLES_DIR
from eval.harness import QA_HOLDOUT, QA_SET
from index.chunker import load_article

TARGET = 50
HOLDOUT_TARGET = 30
MIN_BODY_CHARS = 150
SEED = 42
HOLDOUT_SEED = 7

PROMPT = """You write realistic questions that a Malaysia Airlines passenger would type into a help chat.

Write ONE question that this FAQ article answers.

Rules:
- Do not reuse the article's title wording. Paraphrase it the way a real passenger would ask.
- Keep it under 20 words, plain English, no airline jargon unless a passenger would say it.
- Ask about one specific thing the article covers.
- Output only the question, nothing else.

Article title: {title}
Article body: {body}"""

HOLDOUT_PROMPT = """You are a Malaysia Airlines passenger typing a quick message into a help chat on your phone.

Write ONE message asking about something this FAQ article answers.

Rules:
- At most 12 words. Casual: lowercase is fine, it need not be a full sentence.
- Use your own everyday words. Do not reuse the article title's wording or its product names.
- Ask about one specific thing the article covers.
- Output only the message, nothing else.

Article title: {title}
Article body: {body}"""


def pick_articles(target: int = TARGET, seed: int = SEED, exclude: frozenset = frozenset(),
                  shuffle_topics: bool = False) -> list:
    by_topic = defaultdict(list)
    for path in sorted(ARTICLES_DIR.rglob("*.md")):
        meta, body = load_article(path)
        if len(body) >= MIN_BODY_CHARS and meta["url"] not in exclude:
            by_topic[meta.get("topic") or "Uncategorized"].append((meta, body))

    rng = random.Random(seed)
    for items in by_topic.values():
        rng.shuffle(items)

    # Round-robin across topics so no single topic dominates the set. With fewer questions than
    # topics, shuffling stops the cut from always falling on the alphabetically last topics.
    picked, topics = [], sorted(by_topic)
    if shuffle_topics:
        rng.shuffle(topics)
    while len(picked) < target:
        added = False
        for t in topics:
            if by_topic[t]:
                picked.append(by_topic[t].pop())
                added = True
                if len(picked) == target:
                    break
        if not added:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="write the held-out set instead")
    args = ap.parse_args()

    from eval.harness import timed_call
    from agent.nodes import get_llm

    if args.holdout:
        used = frozenset(r["expected_url"] for r in json.loads(QA_SET.read_text(encoding="utf-8")))
        articles = pick_articles(HOLDOUT_TARGET, HOLDOUT_SEED, used, shuffle_topics=True)
        prompt_tpl, prefix, out = HOLDOUT_PROMPT, "h", QA_HOLDOUT
    else:
        articles, prompt_tpl, prefix, out = pick_articles(), PROMPT, "q", QA_SET

    llm = get_llm()
    rows = []
    for i, (meta, body) in enumerate(articles, 1):
        prompt = prompt_tpl.format(title=meta["title"], body=body[:1200])
        resp, m = timed_call(llm.invoke, prompt)
        if resp is None:
            print(f"{i:>3}. SKIP ({m.error})")
            continue
        question = resp.content.strip().strip('"').splitlines()[0].strip()
        rows.append(
            {
                "id": f"{prefix}{i:03d}",
                "question": question,
                "expected_url": meta["url"],
                "title": meta["title"],
                "topic": meta.get("topic") or "Uncategorized",
            }
        )
        print(f"{i:>3}. [{m.wall_s:>5.2f}s] {question}")

    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
