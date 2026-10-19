"""Layer 2: can the local model reliably emit valid Route and Grade JSON?"""

import pytest

from agent import prompts
from agent.nodes import Grade, Route, get_llm
from eval.harness import FAIL, PASS, TIMEOUT, Metrics, record, timed_call

LAYER = "2-structured"
N = 20
MIN_PARSE_RATE = 0.95

ROUTER_MESSAGES = [
    "hello there",
    "how much baggage can I check in economy?",
    "I want to speak to a human agent",
    "what's the weather in Tokyo tomorrow?",
    "can I bring my cat on the plane?",
]

GRADER_CONTEXT = "\n\n".join(
    f"[{i}] {d}"
    for i, d in enumerate(
        [
            "Topic: Baggage\nQuestion: What is the cabin baggage allowance?\n\n"
            "Economy passengers may carry one piece of cabin baggage up to 7kg.",
            "Topic: Enrich\nQuestion: How do I join Enrich?\n\n"
            "Enrol free on the Malaysia Airlines website.",
        ],
        1,
    )
)

ROUTER_PROMPTS = [prompts.ROUTER.format(history="(none)", message=m) for m in ROUTER_MESSAGES]
GRADER_PROMPTS = [
    prompts.GRADER.format(question=q, context=GRADER_CONTEXT)
    for q in ["what is the cabin baggage limit?", "how do I sign up for Enrich?"]
]


def _parse_rate(schema, prompt_pool, label) -> float:
    """include_raw keeps the raw AIMessage (for tokens) and turns a parse failure into data."""
    llm = get_llm().with_structured_output(schema, method="json_schema", include_raw=True)
    ok, metrics = 0, []
    for i in range(N):
        out, m = timed_call(llm.invoke, prompt_pool[i % len(prompt_pool)])
        metrics.append(m)
        if m.timed_out:
            status, note = TIMEOUT, m.error or "timeout"
        elif out is None:
            status, note = FAIL, m.error or "call failed"
        elif out.get("parsing_error") or not isinstance(out.get("parsed"), schema):
            status, note = FAIL, f"parse error: {str(out.get('parsing_error'))[:80]}"
        else:
            status, note, ok = PASS, "", ok + 1
        if status != PASS:
            record(LAYER, f"{label} rep {i + 1}", status, m, note)

    mean_s = sum(m.wall_s for m in metrics) / len(metrics)
    tok = [m.tok_per_s for m in metrics if m.tok_per_s]
    note = f"{ok}/{N} valid | mean {mean_s:.2f}s"
    if tok:
        note += f" | mean {sum(tok) / len(tok):.0f} tok/s"
    record(
        LAYER,
        f"{label} parse rate ({N} reps)",
        PASS if ok / N >= MIN_PARSE_RATE else FAIL,
        Metrics(wall_s=round(sum(m.wall_s for m in metrics), 2)),
        note=note,
        calls=N,
    )
    return ok / N


@pytest.mark.parametrize(
    "schema,prompt_pool,label",
    [(Route, ROUTER_PROMPTS, "Route"), (Grade, GRADER_PROMPTS, "Grade")],
)
def test_parse_rate(schema, prompt_pool, label):
    rate = _parse_rate(schema, prompt_pool, label)
    assert rate >= MIN_PARSE_RATE, f"{label} parsed {rate:.0%}, below {MIN_PARSE_RATE:.0%}"
