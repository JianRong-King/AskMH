"""Layer 4: grader decisions, plus the malformed-output repair paths.

The repair half runs with a stubbed LLM, so it is instant and fully deterministic -
it exercises the guards in grade() rather than the model.
"""

import pytest
from langchain_core.documents import Document

from agent import nodes
from agent.nodes import Grade
from eval.harness import FAIL, PASS, TIMEOUT, Metrics, record, timed_call

LAYER = "4-grader"

DOCS = [
    Document(
        page_content="Topic: Baggage\nQuestion: What is the cabin baggage allowance?\n\n"
        "Economy passengers may carry one piece of cabin baggage up to 7kg.",
        metadata={"url": "https://example.test/cabin-baggage", "title": "Cabin baggage"},
    ),
    Document(
        page_content="Topic: Enrich\nQuestion: How do I join Enrich?\n\n"
        "Enrolment is free on the Malaysia Airlines website.",
        metadata={"url": "https://example.test/join-enrich", "title": "Join Enrich"},
    ),
]


class _StubStructured:
    def __init__(self, result):
        self.result = result

    def invoke(self, *_args, **_kwargs):
        return self.result


class _StubLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, *_args, **_kwargs):
        return _StubStructured(self.result)


REPAIR_CASES = [
    (
        "answer-with-no-relevant-ids",
        Grade(decision="answer", relevant=[]),
        "none",
        "claims an answer but cites nothing",
    ),
    (
        "clarify-with-blank-question",
        Grade(decision="clarify", relevant=[1], clarifying_question="   "),
        "answer",
        "clarify with no question to ask",
    ),
    (
        "out-of-range-ids",
        Grade(decision="answer", relevant=[99, 0, -1]),
        "none",
        "all ids outside 1..len(docs)",
    ),
    (
        "duplicate-ids",
        Grade(decision="answer", relevant=[1, 1, 2]),
        "answer",
        "duplicates collapse to 2 docs",
    ),
]


@pytest.mark.parametrize(
    "case_id,fake,expected,why", REPAIR_CASES, ids=[c[0] for c in REPAIR_CASES]
)
def test_repair_paths(case_id, fake, expected, why, monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda: _StubLLM(fake))
    out, m = timed_call(nodes.grade, {"question": "what is the cabin baggage limit?", "docs": DOCS})
    assert out is not None, f"grade() raised: {m.error}"

    actual = out["decision"]
    ok = actual == expected
    if case_id == "duplicate-ids" and ok:
        ok = len(out["docs"]) == 2
    record(LAYER, f"repair: {case_id}", PASS if ok else FAIL, Metrics(wall_s=m.wall_s),
           f"{why} -> {actual}", calls=0)  # stubbed LLM
    assert ok, f"{why}: expected {expected}, got {actual} (docs={len(out['docs'])})"


def test_repair_clarify_emits_question(monkeypatch):
    """A well-formed clarify must survive and surface its question to the user."""
    fake = Grade(decision="clarify", relevant=[1], clarifying_question="Which route are you flying?")
    monkeypatch.setattr(nodes, "get_llm", lambda: _StubLLM(fake))
    out, m = timed_call(nodes.grade, {"question": "how much is extra baggage?", "docs": DOCS})
    assert out is not None

    ok = out["decision"] == "clarify" and out["messages"][0].content == "Which route are you flying?"
    record(LAYER, "repair: clarify-passthrough", PASS if ok else FAIL, Metrics(wall_s=m.wall_s),
           out["decision"], calls=0)
    assert ok


LLM_CASES = [
    ("relevant-docs", "what is the cabin baggage allowance?", DOCS, {"answer", "clarify"}),
    ("irrelevant-docs", "how do I rent a car in Paris?", DOCS, {"none"}),
]


@pytest.mark.parametrize(
    "case_id,question,docs,acceptable", LLM_CASES, ids=[c[0] for c in LLM_CASES]
)
def test_grader_decisions(case_id, question, docs, acceptable):
    out, m = timed_call(nodes.grade, {"question": question, "docs": docs})
    if out is None:
        record(LAYER, case_id, TIMEOUT if m.timed_out else FAIL, m, m.error or "no result")
        pytest.fail(f"grade call failed: {m.error}")

    actual = out["decision"]
    ok = actual in acceptable
    record(LAYER, case_id, PASS if ok else FAIL, m,
           f"got {actual}, wanted one of {sorted(acceptable)}" if not ok else actual)
    assert ok, f"{question!r}: got {actual}, expected one of {sorted(acceptable)}"
