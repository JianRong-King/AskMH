"""Layer 1: retrieval quality. No LLM involved."""

import time

from agent.retriever import get_bm25, get_retriever, get_vectorstore, retrieve
from eval.harness import FAIL, PASS, Metrics, record

LAYER = "1-retrieval"
K = 5
MIN_RECALL_AT_5 = 0.80


def _rank_of(expected_url: str, docs) -> int | None:
    """1-based rank of the first chunk belonging to the expected article."""
    for i, d in enumerate(docs, 1):
        if d.metadata["url"] == expected_url:
            return i
    return None


def _search(mode: str, query: str, k: int):
    if mode == "dense":
        return get_vectorstore().similarity_search(query, k=k)
    if mode == "bm25":
        return get_bm25(k).invoke(query)
    if mode == "hybrid":
        return get_retriever(k).invoke(query)[:k]
    return retrieve(query, k)  # whatever the agent is actually configured to use


def test_agent_recall_per_question(qa_set):
    """Record every question individually, then assert on the aggregate."""
    ranks = []
    for row in qa_set:
        t0 = time.time()
        docs = _search("agent", row["question"], K)
        wall = time.time() - t0
        rank = _rank_of(row["expected_url"], docs)
        ranks.append(rank)
        record(
            LAYER,
            f"{row['id']} {row['question'][:52]}",
            PASS if rank else FAIL,
            Metrics(wall_s=round(wall, 3)),
            note=f"rank {rank}" if rank else f"miss (expected {row['title'][:40]})",
            calls=0,  # embeddings + BM25 only, no LLM
        )

    hits = sum(1 for r in ranks if r)
    recall5 = hits / len(ranks)
    assert recall5 >= MIN_RECALL_AT_5, f"Recall@{K} {recall5:.0%} below floor {MIN_RECALL_AT_5:.0%}"


def test_mode_comparison(qa_set):
    """Dense vs BM25 vs hybrid vs whatever the agent uses, over the same labelled set."""
    summary = {}
    for mode in ("dense", "bm25", "hybrid", "agent"):
        t0 = time.time()
        ranks = [_rank_of(r["expected_url"], _search(mode, r["question"], K)) for r in qa_set]
        wall = time.time() - t0

        n = len(ranks)
        at1 = sum(1 for r in ranks if r == 1) / n
        at3 = sum(1 for r in ranks if r and r <= 3) / n
        at5 = sum(1 for r in ranks if r) / n
        mrr = sum(1 / r for r in ranks if r) / n
        summary[mode] = at5
        record(
            LAYER,
            f"mode={mode} ({n} questions)",
            PASS,
            Metrics(wall_s=round(wall, 2)),
            note=f"R@1 {at1:.0%} | R@3 {at3:.0%} | R@5 {at5:.0%} | MRR {mrr:.3f}",
            calls=0,
        )

    # Guards F1: the configured retrieval must not trail an alternative sitting right there.
    best_alternative = max(summary[m] for m in ("dense", "bm25", "hybrid"))
    assert summary["agent"] >= best_alternative - 0.02, (
        f"agent retrieval ({summary['agent']:.0%}) trails an available mode: {summary}"
    )
