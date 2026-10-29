"""End-to-end benchmark of chat-LLM configurations over the 50-question eval set.

    python -m eval.compare_llms --config gemini-off --run 2      # one configuration, one run
    python -m eval.compare_llms --gemini --run 2                 # both Gemini configs in parallel
    python -m eval.compare_llms --report                         # aggregate every saved run

Each question runs the full LangGraph agent in a fresh thread (router -> retrieve -> grade ->
generate/decline). Recorded per question: the router's rewritten question, retrieved and cited
article URLs, rank of the expected article in retrieval, decision, outcome, per-node wall time,
total wall time, and token usage including thinking tokens.

Rows are written to eval/results/bench/<config>_run<n>.json (gitignored, includes full replies).
`--export DIR` copies them without reply text, for committing alongside the write-up.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Not imported from eval.harness: that imports config, which would freeze CHAT_PROVIDER and
# the thinking budget before run() applies the configuration's environment.
EVAL_DIR = Path(__file__).resolve().parent
QA_SET = EVAL_DIR / "qa_set.json"
BENCH_DIR = EVAL_DIR / "results" / "bench"

CONFIGS = {
    "gemini-off": {"CHAT_PROVIDER": "gemini", "CHAT_MODEL": "gemini-2.5-flash", "GEMINI_THINKING_BUDGET": "0",
                   "LLM_RPM": "60"},
    "gemini-dynamic": {"CHAT_PROVIDER": "gemini", "CHAT_MODEL": "gemini-2.5-flash", "GEMINI_THINKING_BUDGET": "-1",
                       "LLM_RPM": "60"},
    "qwen": {"CHAT_PROVIDER": "ollama"},
}


def run(name: str, run_no: int) -> None:
    os.environ.update(CONFIGS[name], LANGSMITH_TRACING="false")

    from langchain_core.callbacks import get_usage_metadata_callback
    from langchain_core.messages import AIMessage, HumanMessage

    import config
    from agent.graph import build_graph
    from app.metrics import outcome_of

    assert config.CHAT_PROVIDER == CONFIGS[name]["CHAT_PROVIDER"], "environment not applied"
    meta = {"config": name, "run": run_no, "provider": config.CHAT_PROVIDER, "model": config.CHAT_MODEL,
            "thinking_budget": config.GEMINI_THINKING_BUDGET if config.CHAT_PROVIDER == "gemini" else None,
            "llm_rpm": config.LLM_RPM, "retrieve_k": config.RETRIEVE_K,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print(f"[{name} r{run_no}] {meta['provider']}:{meta['model']}", flush=True)

    graph = build_graph()
    rows = []
    for qa in json.loads(QA_SET.read_text(encoding="utf-8")):
        config_ = {"configurable": {"thread_id": uuid.uuid4().hex}}
        row = {"id": qa["id"], "question": qa["question"], "topic": qa["topic"], "expected_url": qa["expected_url"]}
        nodes, retrieved = [], []
        t0 = last = time.time()
        try:
            with get_usage_metadata_callback() as cb:
                for update in graph.stream({"messages": [HumanMessage(qa["question"])]}, config_, stream_mode="updates"):
                    now = time.time()
                    for node, value in update.items():
                        nodes.append([node, round(now - last, 3)])
                        if node == "retrieve":
                            retrieved = [d.metadata["url"] for d in value["docs"]]
                    last = now
            state = graph.get_state(config_).values
            replies = [m.content for m in state["messages"][1:] if isinstance(m, AIMessage)]
            outcome = outcome_of(state, len(replies))
            usage = list(cb.usage_metadata.values())
            sources = state.get("sources", []) if outcome == "answered" else []
            row.update({
                "wall_s": round(time.time() - t0, 3),
                "nodes": nodes,
                "route": state.get("route", ""),
                "rewritten_question": state.get("question", ""),
                "decision": state.get("decision", ""),
                "grader_raw": state.get("grader_raw", ""),
                "outcome": outcome,
                "retrieved": retrieved,
                "expected_rank": retrieved.index(qa["expected_url"]) + 1 if qa["expected_url"] in retrieved else None,
                "graded_relevant": [d.metadata["url"] for d in state.get("docs", [])],
                "sources": sources,
                "cited_expected": qa["expected_url"] in sources,
                "llm_calls": sum(n in ("router", "grade", "generate") for n, _ in nodes),
                "tokens_in": sum(u.get("input_tokens", 0) for u in usage),
                "tokens_out": sum(u.get("output_tokens", 0) for u in usage),
                "thinking_tokens": sum(u.get("output_token_details", {}).get("reasoning", 0) for u in usage),
                "reply": replies[-1] if replies else "",
            })
        except Exception as e:
            row.update({"wall_s": round(time.time() - t0, 3), "nodes": nodes, "outcome": "error",
                        "error": f"{type(e).__name__}: {e}"[:500]})
        rows.append(row)
        print(f"[{name} r{run_no}] {qa['id']} {row['outcome']:9} {row['wall_s']:6.2f}s", flush=True)

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / f"{name}_run{run_no}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")


def load_runs() -> dict[tuple[str, int], dict]:
    runs = {}
    for f in sorted(BENCH_DIR.glob("*_run*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        runs[(data["meta"]["config"], data["meta"]["run"])] = data
    return runs


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 2)


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["outcome"] != "error"]
    # The first question of a run pays the embedding-model (and for Qwen, LLM) load.
    walls = [r["wall_s"] for r in ok[1:]]
    count = lambda o: sum(r["outcome"] == o for r in rows)  # noqa: E731
    node_time = {}
    for r in ok[1:]:
        for node, s in r.get("nodes", []):
            node_time.setdefault(node, []).append(s)
    return {
        "answered": count("answered"), "cited_expected": sum(r.get("cited_expected", False) for r in rows),
        "clarify": count("clarify"), "declined": count("declined"), "handoff": count("handoff"),
        "error": count("error"),
        # None for runs recorded before retrieval was logged (Gemini run 1).
        "retrieval_hit": sum(r.get("expected_rank") is not None for r in ok) if all("retrieved" in r for r in ok) else None,
        "mean_s": round(statistics.mean(walls), 2), "median_s": pct(walls, 0.5),
        "p90_s": pct(walls, 0.9), "max_s": round(max(walls), 2),
        "tokens_in": sum(r["tokens_in"] for r in ok), "tokens_out": sum(r["tokens_out"] for r in ok),
        "thinking_tokens": sum(r["thinking_tokens"] for r in ok),
        "node_median_s": {n: pct(v, 0.5) for n, v in node_time.items()},
    }


def report() -> None:
    runs = load_runs()
    out = {f"{c}_run{n}": summarize(d["rows"]) for (c, n), d in runs.items()}
    out_path = BENCH_DIR / "summary.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def export(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for (c, n), data in load_runs().items():
        rows = [{k: v for k, v in r.items() if k != "reply"} for r in data["rows"]]
        (dest / f"{c}_run{n}.json").write_text(
            json.dumps({"meta": data["meta"], "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"exported to {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS))
    p.add_argument("--gemini", action="store_true", help="run both Gemini configs in parallel")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--report", action="store_true")
    p.add_argument("--export", type=Path)
    a = p.parse_args()
    if a.report:
        report()
    elif a.export:
        export(a.export)
    elif a.gemini:
        procs = [subprocess.Popen([sys.executable, "-u", "-m", "eval.compare_llms", "--config", c, "--run", str(a.run)])
                 for c in ("gemini-off", "gemini-dynamic")]
        sys.exit(1 if any(pr.wait() for pr in procs) else 0)
    elif a.config:
        run(a.config, a.run)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
