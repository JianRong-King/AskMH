"""Doc2query: LLM-written passenger questions per article, embedded alongside its chunks.

    python -m index.doc2query               # fill data/doc2query.json for new or changed articles
    python -m index.doc2query --limit 20    # try a few first
    python -m index.doc2query --force       # regenerate everything
    python -m index.build_index             # then rebuild the index to use them

AskMH articles are short and passengers paraphrase, so a question often shares few words with
the article that answers it ("book a flight with points" vs "Travel Award"; eval/FINDINGS.md F5).
The questions only change what is embedded: the grader and generator still see the article text.

Always uses Gemini 2.5 Flash with thinking off, whatever CHAT_PROVIDER says, so it needs
GOOGLE_API_KEY. Results are saved after every batch, so an interrupted run resumes.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel

N_QUESTIONS = 4
BATCH = 10
BODY_CHARS = 2000
MAX_OUTPUT_TOKENS = 4096

PROMPT = """You help a search engine match passenger questions to Malaysia Airlines FAQ articles.

For each article below, write {n} different questions that a passenger might type into a help chat and that the article answers.

Rules:
- Write the way passengers talk. At least two questions must avoid the title's wording and the article's product or programme names, using everyday words for the same thing.
- Cover different things the article answers, or ask its main point in different ways.
- Only ask what the article actually answers. Under 20 words each, plain English.
- Return one item per article, with the article's number as its id.

{articles}"""


class ArticleQuestions(BaseModel):
    id: int
    questions: list[str]


class BatchQuestions(BaseModel):
    items: list[ArticleQuestions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N articles still to do")
    ap.add_argument("--force", action="store_true", help="regenerate articles already done")
    ap.add_argument("--batch", type=int, default=BATCH, help="articles per LLM call")
    ap.add_argument("--workers", type=int, default=4, help="concurrent LLM calls")
    ap.add_argument("--rpm", default="60", help="requests per minute (paid tier)")
    args = ap.parse_args()

    # config reads the environment at import, so the provider is set before anything imports it.
    os.environ.update(CHAT_PROVIDER="gemini", CHAT_MODEL="gemini-2.5-flash", GEMINI_THINKING_BUDGET="0",
                      LLM_RPM=args.rpm, LANGSMITH_TRACING="false")

    from agent.nodes import call_llm, get_llm, normalize, usable_rewrite
    from config import ARTICLES_DIR, DOC2QUERY_FILE
    from index.chunker import article_hash, load_article

    cache = json.loads(DOC2QUERY_FILE.read_text(encoding="utf-8")) if DOC2QUERY_FILE.exists() else {}
    todo = []
    for path in sorted(ARTICLES_DIR.rglob("*.md")):
        meta, body = load_article(path)
        h = article_hash(meta, body)
        if args.force or cache.get(meta["url"], {}).get("hash") != h:
            todo.append((meta, body, h))
    todo = todo[: args.limit] if args.limit else todo
    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(todo)} articles to do in {len(batches)} batches ({len(cache)} cached)", flush=True)
    if not todo:
        return

    # Typographic quotes are normalised on the way in: Gemini's schema-constrained output once
    # degenerated on U+2019 (eval/benchmarks/2026-10-16-gemini-thinking, q004).
    llm = get_llm().model_copy(update={"max_output_tokens": MAX_OUTPUT_TOKENS})
    structured = llm.with_structured_output(BatchQuestions, include_raw=True)

    def run(batch):
        articles = "\n\n".join(
            f"[{i}] Title: {normalize(meta['title'])}\nTopic: {' > '.join(meta.get('topic_path') or [])}\n"
            f"{normalize(body[:BODY_CHARS])}"
            for i, (meta, body, _) in enumerate(batch, 1)
        )
        return call_llm(structured, PROMPT.format(n=N_QUESTIONS, articles=articles))

    t0, done, short, failed, tokens_in, tokens_out = time.time(), 0, 0, 0, 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run, b): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad batch must not lose the rest
                failed += len(batch)
                print(f"  batch failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
                continue
            usage = result["raw"].usage_metadata or {}
            tokens_in += usage.get("input_tokens", 0)
            tokens_out += usage.get("output_tokens", 0)
            out = result["parsed"]
            for item in out.items if out else []:
                if not 1 <= item.id <= len(batch):
                    continue
                meta, _, h = batch[item.id - 1]
                qs = [normalize(q).strip() for q in item.questions if usable_rewrite(q)][:N_QUESTIONS]
                if qs:
                    cache[meta["url"]] = {"hash": h, "questions": qs}
                    done += 1
                    short += len(qs) < N_QUESTIONS
            DOC2QUERY_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"  {done}/{len(todo)} articles, {time.time() - t0:.0f}s", flush=True)

    print(f"done: {done} articles ({short} with fewer than {N_QUESTIONS} questions), "
          f"{len(todo) - done} missing ({failed} in failed batches; rerun to retry), "
          f"{time.time() - t0:.0f}s, tokens in/out {tokens_in}/{tokens_out}")


if __name__ == "__main__":
    main()
