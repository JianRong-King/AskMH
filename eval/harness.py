"""Timing and metrics harness shared by every test layer."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from config import CHROMA_DIR, CHUNKS_FILE, LLM_TIMEOUT

RESULTS_DIR = Path(__file__).resolve().parent / "results"
QA_SET = Path(__file__).resolve().parent / "qa_set.json"

# A warm call is ~0.6s; a cold model load is ~35s. Anything above this in
# load_duration means Ollama evicted the model and the timing is not comparable.
COLD_LOAD_THRESHOLD_S = 1.0

PASS, FAIL, TIMEOUT = "PASS", "FAIL", "TIMEOUT"


@dataclass
class Metrics:
    wall_s: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    tok_per_s: float | None = None
    load_s: float = 0.0
    timed_out: bool = False
    error: str | None = None

    @property
    def cold_reload(self) -> bool:
        return self.load_s > COLD_LOAD_THRESHOLD_S


@dataclass
class Row:
    layer: str
    case: str
    status: str
    metrics: Metrics
    note: str = ""
    calls: int = 1  # >1 when one case batches several LLM calls, so per-call latency stays honest


RESULTS: list[Row] = []


def record(
    layer: str,
    case: str,
    status: str,
    metrics: Metrics | None = None,
    note: str = "",
    calls: int = 1,
) -> None:
    RESULTS.append(Row(layer, case, status, metrics or Metrics(), note, calls))


def _metadata_of(obj) -> dict:
    """Pull Ollama's response metadata out of whatever shape the call returned."""
    if isinstance(obj, dict) and "raw" in obj:
        obj = obj["raw"]
    return getattr(obj, "response_metadata", None) or {}


def extract_metrics(obj, wall_s: float) -> Metrics:
    md = _metadata_of(obj)
    tokens_out = md.get("eval_count")
    eval_ns = md.get("eval_duration") or 0
    tok_per_s = round(tokens_out / (eval_ns / 1e9), 1) if tokens_out and eval_ns else None
    return Metrics(
        wall_s=round(wall_s, 2),
        tokens_in=md.get("prompt_eval_count"),
        tokens_out=tokens_out,
        tok_per_s=tok_per_s,
        load_s=round((md.get("load_duration") or 0) / 1e9, 2),
    )


def timed_call(fn, *args, **kwargs) -> tuple[object | None, Metrics]:
    """Run fn, returning (result, metrics). A transport timeout is not an exception here.

    The timeout is enforced by httpx inside ChatOllama (client_kwargs), because a
    thread-based guard cannot interrupt a blocked socket.
    """
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
    except httpx.TimeoutException as e:
        return None, Metrics(wall_s=round(time.time() - t0, 2), timed_out=True, error=str(e)[:200])
    except Exception as e:  # noqa: BLE001 - a failing case must not abort the layer
        msg = f"{type(e).__name__}: {e}"
        timed_out = "timeout" in msg.lower() or "timed out" in msg.lower()
        return None, Metrics(wall_s=round(time.time() - t0, 2), timed_out=timed_out, error=msg[:200])
    return result, extract_metrics(result, time.time() - t0)


def index_fingerprint() -> dict:
    """Identify the index so a mid-campaign rebuild cannot silently skew results."""
    chunks = sum(1 for _ in CHUNKS_FILE.open(encoding="utf-8"))
    import sqlite3

    db = CHROMA_DIR / "chroma.sqlite3"
    with sqlite3.connect(db) as con:
        embeddings = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    return {"chunks_jsonl": chunks, "chroma_embeddings": embeddings}


def load_qa_set() -> list[dict]:
    with QA_SET.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, dash="-"):
    return dash if v is None else v


def write_results(extra: dict | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": index_fingerprint(),
        "llm_timeout_s": LLM_TIMEOUT,
        "summary": summarize(),
        "extra": extra or {},
        "rows": [{**asdict(r), "metrics": asdict(r.metrics)} for r in RESULTS],
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "| Layer | Case | Status | Duration (s) | Tokens in/out | Tok/s | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in RESULTS:
        m = r.metrics
        tok = f"{_fmt(m.tokens_in)}/{_fmt(m.tokens_out)}" if m.tokens_in or m.tokens_out else "-"
        note = r.note + (" [COLD RELOAD]" if m.cold_reload else "")
        lines.append(
            f"| {r.layer} | {r.case} | {r.status} | {m.wall_s:.2f} | {tok} | {_fmt(m.tok_per_s)} | {note} |"
        )
    md = "\n".join(lines) + "\n\n" + summary_text()
    (RESULTS_DIR / "results.md").write_text(md, encoding="utf-8")
    return RESULTS_DIR / "results.md"


def summarize() -> dict:
    by_layer: dict[str, dict] = {}
    for r in RESULTS:
        s = by_layer.setdefault(
            r.layer, {"pass": 0, "fail": 0, "timeout": 0, "total_s": 0.0, "n": 0, "calls": 0}
        )
        s[r.status.lower()] = s.get(r.status.lower(), 0) + 1
        s["total_s"] += r.metrics.wall_s
        s["n"] += 1
        s["calls"] += r.calls
    for s in by_layer.values():
        s["total_s"] = round(s["total_s"], 2)
        s["mean_s"] = round(s["total_s"] / s["n"], 2) if s["n"] else 0.0
        s["per_call_s"] = round(s["total_s"] / s["calls"], 2) if s["calls"] else 0.0
        s["fail_rate"] = round((s["fail"] + s["timeout"]) / s["n"], 3) if s["n"] else 0.0
    return by_layer


def summary_text() -> str:
    by_layer = summarize()
    if not by_layer:
        return "_No results recorded._"

    # Rank by failure RATE, not raw count: 6 misses out of 50 is healthier than 1 of 3.
    worst_correctness = max(by_layer.items(), key=lambda kv: (kv[1]["fail_rate"], kv[1]["fail"]))
    measured = {k: v for k, v in by_layer.items() if k != "0-setup"} or by_layer
    worst_latency = max(measured.items(), key=lambda kv: kv[1]["per_call_s"])

    out = [
        "## Summary",
        "",
        "| Layer | Pass | Fail | Timeout | Fail rate | Total s | s/case | s/LLM call |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in by_layer.items():
        out.append(
            f"| {name} | {s['pass']} | {s['fail']} | {s['timeout']} | {s['fail_rate']:.0%} | "
            f"{s['total_s']} | {s['mean_s']} | {s['per_call_s']} |"
        )

    cname, cs = worst_correctness
    lname, ls = worst_latency
    out += [
        "",
        (
            f"- **Correctness bottleneck**: {cname} - {cs['fail_rate']:.0%} of its cases failed "
            f"({cs['fail']} fail, {cs['timeout']} timeout of {cs['n']})"
        )
        if cs["fail"] or cs["timeout"]
        else "- **Correctness bottleneck**: none, every layer passed.",
        (
            f"- **Latency bottleneck**: {lname} - {ls['per_call_s']}s per LLM call "
            f"({ls['total_s']}s total over {ls['calls']} calls)"
        ),
    ]
    return "\n".join(out)
