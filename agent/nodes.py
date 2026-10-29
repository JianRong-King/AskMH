import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing_extensions import TypedDict

from agent import prompts
from agent.retriever import retrieve as hybrid_retrieve
from config import (
    CHAT_MODEL,
    CHAT_PROVIDER,
    GEMINI_THINKING_BUDGET,
    LLM_RPM,
    LLM_SEED,
    LLM_TIMEOUT,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
)

MYT = timezone(timedelta(hours=8))
MAX_FAILURES = 2
HISTORY_TURNS = 6

# The router returns a label and one sentence, so this only bites on a runaway generation.
# Gemini once spent 21k-66k tokens and up to 4 minutes on one routing call
# (eval/benchmarks/2026-10-16-gemini-thinking, q004).
ROUTER_MAX_TOKENS = 256

# That runaway was triggered by U+2019 in "Traveller’s": Gemini's schema-constrained JSON stopped
# at the apostrophe and emitted whitespace until the output limit. Phone keyboards insert
# typographic quotes by default, so they are mapped to ASCII before any prompt sees them.
_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...", "\u00a0": " ",
})
_WHITESPACE_RUN = re.compile(r"[\t\r\n]{3,}| {4,}")


def normalize(text: str) -> str:
    return text.translate(_TYPOGRAPHY)


def usable_rewrite(rewrite: str) -> bool:
    """A standalone question is one short sentence; whitespace runs mean the model degenerated.
    Checked before stripping: the observed runaway was trailing whitespace after a cut-off question."""
    return bool(rewrite.strip()) and not _WHITESPACE_RUN.search(rewrite)


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    question: str
    docs: list[Document]
    decision: str
    grader_raw: str
    failures: int
    sources: list[str]


class Route(BaseModel):
    category: Literal["greeting", "human", "faq", "out_of_scope"]
    question: str = Field(description="The latest message rewritten as a standalone question")


class Grade(BaseModel):
    decision: Literal["answer", "clarify", "none"]
    relevant: list[int] = Field(default_factory=list, description="Numbers of the relevant articles")
    clarifying_question: str = ""


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama | ChatGoogleGenerativeAI:
    if CHAT_PROVIDER == "gemini":
        limiter = InMemoryRateLimiter(requests_per_second=LLM_RPM / 60, max_bucket_size=1)
        # max_retries=1 turns off the SDK's own retries (0 means "SDK default"), which would
        # fire immediately and bypass the rate limiter. call_llm retries through it instead.
        return ChatGoogleGenerativeAI(
            model=CHAT_MODEL, temperature=0, thinking_budget=GEMINI_THINKING_BUDGET,
            max_retries=1, rate_limiter=limiter,
        )
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        reasoning=False,
        seed=LLM_SEED,
        keep_alive=OLLAMA_KEEP_ALIVE,
        client_kwargs={"timeout": LLM_TIMEOUT},
    )


@lru_cache(maxsize=1)
def get_router_llm() -> ChatOllama | ChatGoogleGenerativeAI:
    """The chat model with output capped; for Gemini also without thinking, whose tokens would
    count against the cap. Shares the rate limiter with get_llm()."""
    if CHAT_PROVIDER == "gemini":
        return get_llm().model_copy(update={"max_output_tokens": ROUTER_MAX_TOKENS, "thinking_budget": 0})
    return get_llm().model_copy(update={"num_predict": ROUTER_MAX_TOKENS})


_exponential = wait_exponential(multiplier=3, max=30)


def _backoff(state) -> float:
    # A 429 means the per-minute quota is spent and Gemini asks for ~15 s; retrying sooner
    # only fails again and counts against the quota.
    if isinstance(state.outcome.exception(), GoogleRateLimitError):
        return 20.0
    return _exponential(state)


@retry(
    retry=retry_if_exception_type((GoogleAPIError, GoogleRateLimitError)),
    stop=stop_after_attempt(4),
    wait=_backoff,
    reraise=True,
)
def call_llm(runnable, prompt):
    return runnable.invoke(prompt)


def format_history(messages: list[AnyMessage]) -> str:
    lines = [
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {normalize(m.content)}"
        for m in messages[-HISTORY_TURNS:]
    ]
    return "\n".join(lines) or "(none)"


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def router(state: State) -> State:
    *history, latest = state["messages"]
    message = normalize(latest.content)
    llm = get_router_llm().with_structured_output(Route, method="json_schema")
    try:
        route = call_llm(llm, prompts.ROUTER.format(history=format_history(history), message=message))
    except ValueError:
        # Unparseable output, e.g. JSON cut off by ROUTER_MAX_TOKENS. Treat it as an FAQ question:
        # the grader still declines anything the knowledge base cannot answer.
        return {"route": "faq", "question": message, "decision": ""}
    question = route.question if usable_rewrite(route.question) else message
    return {"route": route.category, "question": question, "decision": ""}


def greet(state: State) -> State:
    return {"messages": [AIMessage(prompts.GREETING)]}


def retrieve(state: State) -> State:
    return {"docs": hybrid_retrieve(state["question"])}


def grade(state: State) -> State:
    docs = state["docs"]
    llm = get_llm().with_structured_output(Grade, method="json_schema")
    result = call_llm(llm, prompts.GRADER.format(question=state["question"], context=format_docs(docs)))
    relevant = [docs[i - 1] for i in dict.fromkeys(result.relevant) if 1 <= i <= len(docs)]
    decision = result.decision
    if decision != "none" and not relevant:
        decision = "none"
    if decision == "clarify" and not result.clarifying_question.strip():
        decision = "answer"
    update: State = {"decision": decision, "grader_raw": result.decision, "docs": relevant}
    if decision == "clarify":
        update["messages"] = [AIMessage(result.clarifying_question.strip())]
    return update


def _is_refusal(answer: str) -> bool:
    """True only when the whole reply is the refusal line, not when it merely mentions a gap."""
    normalise = lambda s: s.strip().rstrip(".").casefold()  # noqa: E731
    return normalise(answer) == normalise(prompts.NO_ANSWER)


def generate(state: State) -> State:
    docs = state["docs"]
    answer = call_llm(get_llm(), [
        SystemMessage(prompts.GENERATE.format(context=format_docs(docs), no_answer=prompts.NO_ANSWER)),
        HumanMessage(state["question"]),
    ]).text.strip()

    # The grader thought these articles sufficed but the generator disagreed. Citing sources
    # under a refusal would look answered to the user and to the metrics, so count a failure.
    if _is_refusal(answer):
        return {
            "messages": [AIMessage(prompts.DECLINE_NONE)],
            "sources": [],
            "failures": state.get("failures", 0) + 1,
        }

    sources = list(dict.fromkeys(d.metadata["url"] for d in docs))
    links = "\n".join(f"- {url}" for url in sources)
    return {"messages": [AIMessage(f"{answer}\n\nSources:\n{links}")], "sources": sources, "failures": 0}


def decline(state: State) -> State:
    text = prompts.DECLINE_OUT_OF_SCOPE if state["route"] == "out_of_scope" else prompts.DECLINE_NONE
    return {"messages": [AIMessage(text)], "failures": state.get("failures", 0) + 1, "sources": []}


def handoff(state: State) -> State:
    now = datetime.now(MYT)
    text = prompts.HANDOFF_OPEN if 7 <= now.hour < 23 else prompts.HANDOFF_CLOSED.format(now=now.strftime("%I:%M %p"))
    return {"messages": [AIMessage(text)], "failures": 0}


def after_router(state: State) -> str:
    return {"greeting": "greet", "human": "handoff", "faq": "retrieve"}.get(state["route"], "decline")


def after_grade(state: State) -> str:
    return {"answer": "generate", "none": "decline"}.get(state["decision"], "end")


def after_failure(state: State) -> str:
    """Shared by decline and generate: escalate once the user has been let down twice."""
    return "handoff" if state.get("failures", 0) >= MAX_FAILURES else "end"
