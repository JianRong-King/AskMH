"""Layer 5: is the generated answer grounded in the retrieved context, and correctly cited?"""

import re

import pytest
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from agent import nodes
from eval.harness import FAIL, PASS, TIMEOUT, record, timed_call

LAYER = "5-generation"

ANSWERABLE = [
    Document(
        page_content="Topic: Baggage\nQuestion: What is the cabin baggage allowance?\n\n"
        "Economy passengers may carry one piece of cabin baggage up to 7kg. "
        "Business Class passengers may carry two pieces totalling 14kg.",
        metadata={"url": "https://example.test/cabin-baggage", "title": "Cabin baggage"},
    )
]

UNANSWERABLE = [
    Document(
        page_content="Topic: Enrich\nQuestion: How do I join Enrich?\n\n"
        "Enrolment is free on the Malaysia Airlines website.",
        metadata={"url": "https://example.test/join-enrich", "title": "Join Enrich"},
    )
]

URL_RE = re.compile(r"https?://\S+")


class Verdict(BaseModel):
    grounded: bool = Field(description="True only if every factual claim is supported by the context")
    reason: str = Field(description="One short sentence")


JUDGE = """You are checking whether an answer stayed within its source material.

Context given to the assistant:
{context}

Assistant's answer:
{answer}

Is every factual claim in the answer supported by the context? Ignore the Sources list.
Answer grounded=false if the answer asserts any specific fact not present in the context.

Saying that the articles do NOT cover something is not a factual claim - it is an admission
of a gap, and it keeps the answer grounded. Declining to answer is fully grounded."""


def _judge(context: str, answer: str) -> Verdict | None:
    llm = nodes.get_llm().with_structured_output(Verdict, method="json_schema")
    verdict, _ = timed_call(llm.invoke, JUDGE.format(context=context, answer=answer))
    return verdict


def test_citations_come_from_retrieved_docs():
    out, m = timed_call(
        nodes.generate, {"question": "what is the cabin baggage allowance?", "docs": ANSWERABLE}
    )
    if out is None:
        record(LAYER, "citation match", TIMEOUT if m.timed_out else FAIL, m, m.error or "no result")
        pytest.fail(f"generate failed: {m.error}")

    allowed = {d.metadata["url"] for d in ANSWERABLE}
    cited = set(URL_RE.findall(out["messages"][0].content))
    stray = cited - allowed
    ok = not stray and out["sources"] == list(allowed)
    record(LAYER, "citation match", PASS if ok else FAIL, m,
           f"stray urls: {stray}" if stray else f"{len(cited)} cited, all from context")
    assert ok, f"answer cited URLs not in the retrieved set: {stray}"


def test_answer_is_grounded():
    out, m = timed_call(
        nodes.generate, {"question": "what is the cabin baggage allowance?", "docs": ANSWERABLE}
    )
    if out is None:
        record(LAYER, "groundedness", TIMEOUT if m.timed_out else FAIL, m, m.error or "no result")
        pytest.fail(f"generate failed: {m.error}")

    answer = out["messages"][0].content.split("\n\nSources:")[0]
    verdict = _judge(ANSWERABLE[0].page_content, answer)
    ok = verdict is not None and verdict.grounded
    record(LAYER, "groundedness", PASS if ok else FAIL, m,
           verdict.reason[:80] if verdict else "judge failed", calls=2)  # generate + judge
    assert ok, f"answer not grounded: {verdict.reason if verdict else 'judge failed'}"


def test_does_not_invent_when_context_lacks_answer():
    """Context is about Enrich enrolment; the question is about baggage weight.

    The correct behaviour is to decline, and generate() must turn that refusal into a
    decline with no sources attached - citing articles under a refusal would read as
    answered to both the user and the metrics.
    """
    out, m = timed_call(
        nodes.generate, {"question": "what is the cabin baggage weight limit?", "docs": UNANSWERABLE}
    )
    if out is None:
        record(LAYER, "no-fabrication", TIMEOUT if m.timed_out else FAIL, m, m.error or "no result")
        pytest.fail(f"generate failed: {m.error}")

    reply = out["messages"][0].content
    invented_kg = re.search(r"\b\d+\s?kg\b", reply, re.I)
    # A refusal must also count as a failure, so repeated refusals still reach handoff.
    declined = out["sources"] == [] and "Sources:" not in reply and out.get("failures") == 1

    if invented_kg:
        status, note = FAIL, f"invented {invented_kg.group()}"
    elif declined:
        status, note = PASS, "declined, no sources attached"
    else:
        verdict = _judge(UNANSWERABLE[0].page_content, reply)
        status = PASS if (verdict is None or verdict.grounded) else FAIL
        note = "answered without inventing figures" if status == PASS else "judge: ungrounded"

    record(LAYER, "no-fabrication", status, m, note, calls=2)
    assert status == PASS, f"expected a clean decline, got: {reply[:200]!r}"
