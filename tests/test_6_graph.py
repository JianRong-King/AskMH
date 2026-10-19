"""Layer 6: the full graph, including the failure paths left untested in Phase 3/4."""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from eval.harness import FAIL, PASS, TIMEOUT, Metrics, record, timed_call

LAYER = "6-graph"

HANDOFF_MARKER = "7:00 am-11:00 pm"
DECLINE_MARKER = "Sorry, I c"  # covers both decline strings
# Airline-flavoured so the router sends them down the FAQ path, but absent from any FAQ.
UNANSWERABLE = [
    "What is the first name of the pilot flying MH2847 next Tuesday?",
    "How many rivets are in the left wing of your A350?",
]


def _turn(graph, config, text):
    """Run one user turn, returning (state, new AI message texts, metrics)."""
    before = len(graph.get_state(config).values.get("messages", []))
    state, m = timed_call(graph.invoke, {"messages": [HumanMessage(text)]}, config)
    if state is None:
        return None, [], m
    replies = [x.content for x in state["messages"][before + 1:] if isinstance(x, AIMessage)]
    return state, replies, m


@pytest.fixture
def session():
    graph = build_graph()
    return graph, {"configurable": {"thread_id": str(uuid.uuid4())}}


def _calls_for(state) -> int:
    """Router always runs; an FAQ adds the grader; an answered FAQ adds the generator."""
    if state is None or state.get("route") != "faq":
        return 1
    return 3 if state.get("decision") == "answer" else 2


def _check(case, ok, m, note, state=None):
    record(
        LAYER,
        case,
        PASS if ok else (TIMEOUT if m.timed_out else FAIL),
        Metrics(wall_s=m.wall_s),
        note,
        calls=_calls_for(state),
    )


def test_greeting(session):
    graph, config = session
    state, replies, m = _turn(graph, config, "hi there")
    assert state is not None, f"graph failed: {m.error}"
    ok = state["route"] == "greeting" and replies and "AskMH" in replies[0]
    _check("greeting", ok, m, f"route={state['route']}", state)
    assert ok


def test_happy_path_answers_with_sources(session):
    graph, config = session
    state, replies, m = _turn(graph, config, "How do I know if I am eligible to redeem Enrich Upgrade?")
    assert state is not None, f"graph failed: {m.error}"
    ok = (
        state["route"] == "faq"
        and state["decision"] == "answer"
        and bool(state.get("sources"))
        and "Sources:" in replies[-1]
    )
    _check("happy path (faq -> answer)", ok, m,
           f"decision={state.get('decision')} sources={len(state.get('sources') or [])}", state)
    assert ok, f"expected a cited answer, got decision={state.get('decision')}"


def test_out_of_scope_declines(session):
    graph, config = session
    state, replies, m = _turn(graph, config, "what's the weather in Tokyo tomorrow?")
    assert state is not None, f"graph failed: {m.error}"
    ok = state["route"] == "out_of_scope" and replies and DECLINE_MARKER in replies[-1]
    _check("out-of-scope decline", ok, m, f"route={state['route']} failures={state.get('failures')}", state)
    assert ok, f"expected an out-of-scope decline, got route={state['route']}"


def test_explicit_human_request_hands_off(session):
    graph, config = session
    state, replies, m = _turn(graph, config, "I want to talk to a human agent")
    assert state is not None, f"graph failed: {m.error}"
    ok = state["route"] == "human" and replies and HANDOFF_MARKER in replies[-1]
    _check("explicit human -> handoff", ok, m, f"route={state['route']}", state)
    assert ok, f"expected handoff, got route={state['route']} reply={replies[-1][:80] if replies else None!r}"


def test_ambiguous_question_asks_to_clarify(session):
    graph, config = session
    state, replies, m = _turn(graph, config, "how much is extra baggage?")
    assert state is not None, f"graph failed: {m.error}"
    ok = state.get("decision") == "clarify" and bool(replies)
    _check("ambiguous -> clarify", ok, m, f"decision={state.get('decision')}", state)
    assert ok, f"expected clarify, got decision={state.get('decision')}"


def test_two_failures_then_handoff(session):
    """Loop termination: the second consecutive failure must escalate, not loop forever."""
    graph, config = session

    state1, replies1, m1 = _turn(graph, config, UNANSWERABLE[0])
    assert state1 is not None, f"graph failed: {m1.error}"
    first_declined = bool(replies1) and DECLINE_MARKER in replies1[-1]
    _check("failure 1 declines", first_declined, m1,
           f"failures={state1.get('failures')} decision={state1.get('decision')}", state1)

    state2, replies2, m2 = _turn(graph, config, UNANSWERABLE[1])
    assert state2 is not None, f"graph failed: {m2.error}"
    escalated = any(HANDOFF_MARKER in r for r in replies2)
    _check("failure 2 -> handoff (loop termination)", escalated, m2,
           f"failures={state2.get('failures')} replies={len(replies2)}", state2)

    assert first_declined, f"first unanswerable question should decline, got {replies1}"
    assert escalated, f"second failure should escalate to handoff, got {replies2}"
