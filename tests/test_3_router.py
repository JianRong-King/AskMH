"""Layer 3: does the router pick the right category, and rewrite follow-ups standalone?"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from eval.harness import FAIL, PASS, TIMEOUT, record, timed_call

LAYER = "3-router"

CASES = [
    ("greeting", "hi there", "greeting"),
    ("greeting-thanks", "thanks, that's all", "greeting"),
    ("human", "I want to talk to a human agent", "human"),
    ("human-livechat", "connect me to live chat please", "human"),
    ("faq-baggage", "how much baggage can I check in economy?", "faq"),
    ("faq-pets", "can I bring my cat in the cabin?", "faq"),
    ("out-of-scope", "what's the weather in Tokyo tomorrow?", "out_of_scope"),
    ("out-of-scope-other-airline", "how do I book a Singapore Airlines flight?", "out_of_scope"),
]


@pytest.mark.parametrize("case_id,message,expected", CASES, ids=[c[0] for c in CASES])
def test_router_category(case_id, message, expected):
    state, m = timed_call(nodes.router, {"messages": [HumanMessage(message)]})
    if state is None:
        record(LAYER, case_id, TIMEOUT if m.timed_out else FAIL, m, m.error or "no result")
        pytest.fail(f"router call failed: {m.error}")

    actual = state["route"]
    ok = actual == expected
    record(LAYER, case_id, PASS if ok else FAIL, m,
           f"expected {expected}, got {actual}" if not ok else actual)
    assert ok, f"{message!r}: expected {expected}, got {actual}"


def test_follow_up_rewritten_standalone():
    """'what about business class?' only makes sense against the previous turn."""
    history = [
        HumanMessage("how much baggage can I check in economy?"),
        AIMessage("Economy passengers get 30kg on most routes."),
        HumanMessage("what about business class?"),
    ]
    state, m = timed_call(nodes.router, {"messages": history})
    if state is None:
        record(LAYER, "follow-up rewrite", TIMEOUT if m.timed_out else FAIL, m, m.error or "no result")
        pytest.fail(f"router call failed: {m.error}")

    q = state["question"].lower()
    ok = "business" in q and any(w in q for w in ("baggage", "luggage", "allowance"))
    record(LAYER, "follow-up rewrite", PASS if ok else FAIL, m, state["question"][:70])
    assert ok, f"follow-up not resolved into a standalone question: {state['question']!r}"


def test_curly_apostrophe_routes_cleanly():
    """U+2019 once sent Gemini's router into a whitespace runaway (21k-66k tokens, up to 4 min)."""
    message = "Can I buy MH Traveller’s Trio for a domestic flight from KLIA?"
    state, m = timed_call(nodes.router, {"messages": [HumanMessage(message)]})
    if state is None:
        record(LAYER, "curly apostrophe", TIMEOUT if m.timed_out else FAIL, m, m.error or "no result")
        pytest.fail(f"router call failed: {m.error}")

    ok = state["route"] == "faq" and nodes.usable_rewrite(state["question"]) and "Trio" in state["question"]
    record(LAYER, "curly apostrophe", PASS if ok else FAIL, m, state["question"][:70])
    assert ok, f"got route={state['route']!r}, question={state['question']!r}"


def test_normalize_maps_typography_to_ascii():
    assert nodes.normalize("Traveller’s “Trio” – KLIA… ok") == "Traveller's \"Trio\" - KLIA... ok"


@pytest.mark.parametrize("rewrite,usable", [
    ("Can I buy MH Traveller\n\t\t\t\t\t\t", False),
    ("Can I buy MH Traveller\n\n\n\n", False),
    ("   ", False),
    ("Can I buy MH Traveller's Trio for a domestic flight?", True),
    ("What is the fee?  ", True),
])
def test_usable_rewrite(rewrite, usable):
    assert nodes.usable_rewrite(rewrite) is usable


class _FakeRouterLLM:
    """Stands in for get_router_llm(): returns a fixed Route or raises, without a model call."""

    def __init__(self, result):
        self.result = result

    def with_structured_output(self, *args, **kwargs):
        return self

    def invoke(self, prompt):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_degenerate_rewrite_falls_back_to_message(monkeypatch):
    fake = _FakeRouterLLM(nodes.Route(category="faq", question="Can I buy MH Traveller\n\t\t\t\t\t"))
    monkeypatch.setattr(nodes, "get_router_llm", lambda: fake)
    state = nodes.router({"messages": [HumanMessage("Can I buy MH Traveller’s Trio?")]})
    assert state["route"] == "faq"
    assert state["question"] == "Can I buy MH Traveller's Trio?"


def test_unparseable_router_output_is_treated_as_faq(monkeypatch):
    from langchain_core.exceptions import OutputParserException

    monkeypatch.setattr(nodes, "get_router_llm", lambda: _FakeRouterLLM(OutputParserException("cut off")))
    state = nodes.router({"messages": [HumanMessage("how much baggage can I check?")]})
    assert state == {"route": "faq", "question": "how much baggage can I check?", "decision": ""}
