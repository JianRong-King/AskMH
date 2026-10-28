# Phase 5 findings: four defects, why they happened, and how they were fixed

The Phase 5 test suite was the first time this chatbot was measured rather than spot-checked.
It found four defects. Two of them had been shipping since Phase 2 and Phase 3 without anyone
noticing, because the earlier checks were small hand-picked smoke tests.

This document records, for each defect: what was observed, why it happened, the exact change
that fixed it, and the evidence that the fix worked.

**Environment**: local Ollama `qwen3:8b` (Q4_K_M, digest `500a1f067a9f7826`), RTX 4060 Laptop
8GB, i7-13800H, 63.8GB RAM. `temperature=0`, `seed=42`, frozen index of 1,526 chunks.

**Reproduce**: `python -m eval.run_suite` (about 70 s). Results land in `eval/results/`.

| ID | Defect | Severity | Where it was fixed |
|---|---|---|---|
| F1 | Hybrid retrieval ranked worse than dense alone | Medium | `agent/retriever.py` |
| F2 | Generator invented facts absent from its context | **High** | `agent/prompts.py` (`GENERATE`) |
| F3 | Grader dead-ended instead of asking a clarifying question | Medium | `agent/prompts.py` (`GRADER`) |
| F4 | A refusal was still presented as a cited answer | **High** | `agent/nodes.py`, `agent/graph.py`, `prompts.ROUTER` |

**Before**: 26 pass, 3 fail, 0 timeouts.
**After**: 29 pass, 0 fail, 0 timeouts.

---

# F1 - Hybrid retrieval ranked worse than dense alone

### What happened

`tests/test_1_retrieval.py::test_mode_comparison` failed. Measured over 50 labelled questions:

| mode | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|
| dense only | 76% | 94% | **96%** | **0.848** |
| hybrid (shipped) | 72% | 82% | 88% | 0.781 |
| BM25 only | 58% | 68% | 78% | 0.650 |

The agent was using hybrid. So on 6 questions out of 50, the correct article never entered
the grader's context at all - the pipeline could not possibly answer them, no matter how good
the LLM was.

### Why it happened

Two separate causes.

**1. The Phase 2 evidence was too weak to detect it.** Phase 2 validated retrieval with 10
hand-picked queries and recorded "hybrid ranks every target 1st". With 10 easy queries every
mode looks perfect, so there was nothing to distinguish them. The gap only becomes visible
across 50 questions.

**2. Reciprocal Rank Fusion gives a weak retriever real influence.** `EnsembleRetriever`
combines the two rankings by RRF with `weights=[0.5, 0.5]`. BM25 alone is far weaker here
(78% vs 96%), and lexical search is a poor fit for this corpus: AskMH articles are short, and
passengers paraphrase rather than reuse the article's wording. Averaging a strong ranker with
a weak one at equal weight drags results toward the weak one.

A `dense_weight` sweep confirmed no setting rescued it:

| dense_weight | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | dense only |
|---|---|---|---|---|---|---|
| natural R@5 | 88% | 88% | 88% | 94% | 96% | **96%** |
| natural MRR | 0.781 | 0.752 | 0.777 | 0.786 | 0.813 | **0.848** |

**The counter-argument, which had to be checked.** BM25 was added in Phase 2 specifically for
short exact-term queries - product names like `Platinum Companion`, `MHupgrade`, `MHgiftcard`.
The 50-question set is entirely LLM-paraphrased natural language, which structurally favours
dense embeddings. Judging BM25 on that set alone would have been rigging the comparison.

So a second set of 6 bare product-name queries was tested:

| mode | exact-term R@5 | exact-term MRR |
|---|---|---|
| dense only | **100%** | 0.929 |
| hybrid w=0.5 | **100%** | **1.000** |

BM25 does help here, but far less than Phase 2 assumed: dense already finds **every**
exact-term article inside the top 5. BM25's entire contribution is moving one query of six
from rank 2 to rank 1. That is a real but small gain, paid for with 8 points of Recall@5 on
natural questions - which is how the majority of real users actually type.

### The fix

Retrieve dense-only. Over-fetch and collapse by article, so one long multi-chunk article
cannot occupy several of the five slots handed to the grader.

```python
# agent/retriever.py  -  before
def retrieve(query: str, k: int = RETRIEVE_K) -> list[Document]:
    return get_retriever(k).invoke(query)[:k]

# after
def retrieve(query: str, k: int = RETRIEVE_K) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for doc in get_vectorstore().similarity_search(query, k=k * 3):
        url = doc.metadata["url"]
        if url in seen:
            continue
        seen.add(url)
        out.append(doc)
        if len(out) == k:
            break
    return out
```

BM25 and the ensemble are not deleted - `index/search.py --mode bm25|hybrid` still reaches
them, so the Phase 2 work remains available for comparison.

The test was also rewritten to measure *whatever the agent is actually configured to use*
and assert it does not trail an alternative sitting right there in the codebase. That way the
test guards the decision rather than hard-coding it.

### Verified

Agent retrieval is now R@1 76% / R@3 94% / R@5 96% / MRR 0.848, matching dense and beating
the previous hybrid on every metric. Honest note: the per-article dedupe did not change the
score on this set, because only 20 of 1,483 articles are split into multiple chunks. It is a
safeguard against a plausible failure, not a measured gain.

---

# F2 - The generator invented facts absent from its context

This is the most serious of the four. The product's entire premise is answering strictly from
the AskMH knowledge base.

### What happened

`tests/test_5_generation.py::test_does_not_invent_when_context_lacks_answer` failed.

The generator was handed **one article about Enrich enrolment** and asked an unrelated
question, "what is the cabin baggage weight limit?". It replied:

> The cabin baggage weight limit is 7 kg (15 lb).

Nothing resembling that sentence was in the context. Reproducible 3 times out of 3 with
`seed=42`.

What makes this dangerous is that **7 kg is plausible** - it is roughly the real Malaysia
Airlines limit. The model recalled it from pretraining. Nothing in the reply signals it was
invented, so neither a user nor the metrics could tell. The identical mechanism would produce
a confidently wrong fee for a route where the true answer differs, and that is the case nobody
would catch.

### Why it happened

The old prompt asked politely and lost:

> Answer the passenger's question using only the articles below. [...] If the articles only
> partly answer it, say what they cover and do not guess the rest.

Three weaknesses:

1. **"using only the articles below" is a soft constraint.** It competes against a
   high-confidence pretrained fact and loses. The instruction never says prior knowledge is
   *disallowed as evidence*, only that the articles should be used.
2. **There was no defined way to say "I can't".** The prompt described partial answers but
   gave the model no concrete output for the total-miss case, so answering was the only shape
   available to it.
3. **No rule targeted the actual risk.** The dangerous output is a specific figure. Nothing
   singled out numbers, fees or weights.

### The fix

Rewrite `GENERATE` to deny prior knowledge the status of evidence, ban unsupported specifics,
and name the exact refusal string to emit.

```python
# agent/prompts.py  -  before
GENERATE = """You are AskMH, the Malaysia Airlines FAQ assistant.

Answer the passenger's question using only the articles below. Keep key numbers, conditions
and exceptions. If the articles only partly answer it, say what they cover and do not guess
the rest. Be concise. [...]"""

# after
NO_ANSWER = "The AskMH articles I found don't cover that."

GENERATE = """You are AskMH, the Malaysia Airlines FAQ assistant.

The articles below are your only source. What you remember about Malaysia Airlines from
training is not evidence and must not appear in the answer.

Rules:
- Never state a number, fee, weight, date or condition that does not appear in the articles.
- If the articles do not answer the question, reply with exactly: "{no_answer}" Add nothing else.
- If they answer only part of it, give that part and say plainly what is not covered.
- Keep the key numbers, conditions and exceptions that the articles do give.
- Be concise. [...]"""
```

### Verified

Same input now produces the refusal instead of the invented figure. And on a context that
*does* support an answer, it stays inside the evidence and states the gap rather than filling
it:

> Economy passengers may carry one piece of cabin baggage up to 7kg. Business Class passengers
> may carry two pieces totalling 14kg. **The AskMH articles I found don't cover the allowance
> for premium economy passengers.**

Caveat worth keeping in mind: this is a behavioural fix, not a structural one. A different
model or a careless prompt edit could regress it. `tests/test_5_generation.py` is the guard.

---

# F3 - The grader dead-ended instead of asking a clarifying question

### What happened

`tests/test_6_graph.py::test_ambiguous_question_asks_to_clarify` failed. The question
"how much is extra baggage?" was graded `none`, so the bot declined and counted a failure.

The designed behaviour is `clarify` - the price genuinely depends on route and weight, so one
short follow-up question would make it answerable.

### Why it happened

**Not retrieval.** This was checked directly before blaming the prompt. Retrieval returned
five on-topic articles (extra legroom with 15kg bundle, MHUpgrade interaction, exceeding the
allowance, Enrich allowance). The grader saw relevant material and still chose `none`.

The cause was the prompt presenting three options as a flat list with no precedence:

```
- answer: at least one article answers the question
- clarify: the articles are on topic, but the answer depends on details the user has not given
- none: no article answers the question
```

Read literally, `none` is *true* here - no single article answers the bare question as asked.
`clarify` and `none` both look defensible, and `none` is the simpler judgement, so the model
took it. Nothing told it that `clarify` outranks `none` when the articles are on-topic.

### The fix

Make the precedence explicit and narrow `none` to genuinely off-topic material.

```python
# agent/prompts.py  -  after
Decide, in this order:
1. answer: at least one article answers the question as asked.
2. clarify: the articles are on topic, but the answer depends on a detail the passenger has
   not given (route, cabin class, fare type, weight, Enrich tier, etc.). Write one short
   clarifying question.
3. none: choose this ONLY when the articles are off topic. If they cover the right subject
   and you merely need one more detail from the passenger, choose clarify, not none.
```

### Verified

The same question now grades `clarify` and produces a usable follow-up:

> What is your free baggage allowance (e.g., 30kg, 20kg, etc.) and what is the total weight
> of your baggage?

---

# F4 - A refusal was still presented as a cited answer

**This defect was created by the F2 fix and found immediately after it.** It is the clearest
argument in this document for testing end-to-end and not just per-unit: F2's unit test passed
while the user-visible behaviour was broken.

### What happened

Running the full graph on "how much is extra baggage?" produced:

> The AskMH articles I found don't cover that.
>
> **Sources:**
> - .../What-is-the-free-cabin-baggage-allowance-on-Malaysia-Airlines
> - ...

A refusal, decorated with source links, rendered exactly like a successful answer.

Three consequences:

1. **The user sees a contradiction** - "I don't have this" directly above citations implying
   it was answered from them.
2. **The metrics lie.** `app/main.py` classified the turn by `decision` alone, so
   `decision == "answer"` logged it as `answered` in `data/metrics.db` - inflating the
   resolution rate and attaching CTR links to a non-answer.
3. **Escalation silently broke.** Only `decline` incremented `failures`; `generate` always set
   `failures = 0`. So a user could be refused repeatedly and **never** reach the human handoff,
   defeating the 2-failure rule from Phase 3.

### Why it happened

**Cause A - an assumption in the graph that stopped being true.** `generate()` was written
when reaching it guaranteed an answer, so it unconditionally attached sources and reset the
failure count:

```python
sources = list(dict.fromkeys(d.metadata["url"] for d in docs))
return {"messages": [...], "sources": sources, "failures": 0}
```

F2 gave the generator the ability to refuse, which invalidated that assumption. The grader
decides whether articles *look* sufficient; the generator is the one that discovers they are
not. Nothing propagated that discovery back.

**Cause B - the router silently widened the question.** The rewrite step turned:

> "how much is extra baggage?"  ->  "How much is extra baggage **allowance** for Malaysia Airlines?"

That converts a question about **cost** into one about **allowance**. Retrieval then fetched
allowance articles, and the grader reasonably judged them sufficient - for the question it was
shown, which was no longer the question the passenger asked. The old prompt said only
"rewrite the latest message as a standalone question", with nothing forbidding added
qualifiers.

### The fix

**1. A refusal becomes a decline, not an answer** (`agent/nodes.py`):

```python
def _is_refusal(answer: str) -> bool:
    """True only when the whole reply is the refusal line, not when it merely mentions a gap."""
    normalise = lambda s: s.strip().rstrip(".").casefold()
    return normalise(answer) == normalise(prompts.NO_ANSWER)


def generate(state: State) -> State:
    ...
    if _is_refusal(answer):
        return {
            "messages": [AIMessage(prompts.DECLINE_NONE)],
            "sources": [],
            "failures": state.get("failures", 0) + 1,
        }
```

The whole-string match matters: a *partial* answer that merely mentions a gap ("...don't cover
premium economy") must stay an answer and keep its sources. Only a total refusal converts.

**2. `generate` escalates on the same rule as `decline`** (`agent/graph.py`). `after_decline`
was renamed `after_failure` since it now serves both:

```python
# before
g.add_conditional_edges("decline", nodes.after_decline, {"handoff": "handoff", "end": END})
for name in ["greet", "generate", "handoff"]:
    g.add_edge(name, END)

# after
for name in ["decline", "generate"]:
    g.add_conditional_edges(name, nodes.after_failure, {"handoff": "handoff", "end": END})
for name in ["greet", "handoff"]:
    g.add_edge(name, END)
```

**3. The router may no longer change the question's meaning** (`agent/prompts.py`):

> Resolve pronouns and references only. Keep the passenger's own wording and keep the question
> exactly as broad or as narrow as they asked it. Do not add qualifiers they did not say:
> "how much is extra baggage?" asks about cost, so it must not become "how much is the extra
> baggage allowance?".

**4. The UI stops trusting `decision` alone** (`app/main.py`):

```python
# A grader "answer" plus an empty sources list means the generator refused.
# Counting that as answered would overstate the resolution rate.
if state["decision"] == "answer" and state.get("sources"):
    return "answered"
return "handoff" if n_replies > 1 else "declined"
```

### Verified

A refusal now returns no sources, increments `failures`, and is logged as `declined`. All
seven `outcome_of` cases were checked directly, including the refused-and-escalated one.

---

# Overall result

| | Before | After |
|---|---|---|
| Suite | 26 pass, 3 fail | **29 pass, 0 fail, 0 timeouts** |
| Retrieval R@5 | 88% | **96%** |
| Retrieval MRR | 0.781 | **0.848** |
| Grounded when context is insufficient | invents a figure | declines cleanly |
| Refusal handling | shown as cited answer | decline, counted, escalates |

Determinism was confirmed by running the suite twice: every status and every decision was
identical, with only throughput jitter differing (0.68 vs 0.78 s mean per call). `seed=42` and
`temperature=0` are doing their job.

Latency on this hardware: ~35 s one-off cold model load, then 0.6-0.9 s per warm LLM call. A
three-call FAQ turn is about 2 s; a full browser turn measured 4.8 s, against ~50 s on the
Gemini free tier.

### Still open

- **2 of 50 questions still miss at k=5.** Both are genuinely hard. A reranker is the next
  lever if recall needs to beat 96%.
- **F2 and F3 are prompt-level fixes.** They hold for `qwen3:8b` at `seed=42`; a model swap
  should be treated as invalidating them until the suite is re-run.
- **The exact-term edge BM25 held was given up deliberately** (one query at rank 1 instead of
  rank 2). If short product-name queries later turn out to dominate real traffic, revisit F1
  with `--mode hybrid` and a high `dense_weight`.
- **The 50-question set is LLM-generated.** It is realistic but not real user traffic, and it
  inherits whatever phrasing biases the generating model has.

### What the earlier phases should have done differently

- A 10-query smoke test cannot distinguish retrieval strategies. It validated that the
  pipeline *runs*, and was mistaken for evidence that it works well (F1).
- "Answer only from the context" in a prompt is an intention, not a guarantee, and needs a
  test that deliberately supplies insufficient context (F2).
- When one component gains a new behaviour, the components downstream that assumed it could
  not do that are now wrong. F2's unit test passed while the end-to-end output was broken (F4).

## Later benchmarks

- **2026-10-16, Gemini 2.5 Flash thinking off vs dynamic, with Qwen3-8B as reference.** See
  [`benchmarks/2026-10-16-gemini-thinking/REPORT.md`](benchmarks/2026-10-16-gemini-thinking/REPORT.md).
  - Thinking gives no accuracy gain (36 vs 36–37 correct of 50), about 1.7x the median latency and about
    5x the cost.
  - Gemini resolves 36/50 vs Qwen's 22/50.
  - Gemini's router produces degenerate output on a curly apostrophe (U+2019).
- **2026-10-28, doc2query.** See F5 below: better ranking, same 38/50 end to end.

---

# F5 - Doc2query: generated questions in the embedding

### Why

The two remaining misses, q001 and q040, are vocabulary gaps, not hard questions.

- **q001** asks "use my Enrich Points to book a flight". The site calls this a "Travel Award", and the labelled article ("Where do I redeem my Enrich Points?") never mentions flights. It was not in the top 30.
- **q040** asks "add a new card to my account". "Card" pulls in the Enrich membership-card articles, and the labelled "How do I manage my Saved Cards?" ranked 11th.

### Change

- `index/doc2query.py`: Gemini 2.5 Flash (thinking off) writes 4 passenger-style questions per article, at least two of them avoiding the title's wording and product names.
  - Batches of 10 articles per call, cached in `data/doc2query.json` keyed by a hash of title and body.
  - All 1,483 articles took 176 s, with 162k input and 103k output tokens (about $0.31 at list price).
- `index/chunker.py`: the questions go into chunk metadata. `embedding_text()` puts them after the `Topic`/`Question` header.
- `index/build_index.py`: embeds that augmented text but stores the plain chunk, writing through `chromadb` directly. The grader and generator never see generated text, so it cannot be quoted as fact.

### Measuring it fairly

`eval/qa_set.json` is itself LLM-paraphrased, so generated questions could match it for the wrong reason. A held-out set was built **before** the reindex: `eval/qa_holdout.json`, from `python -m eval.build_qa_set --holdout`.
- 30 short, casual questions, written by Qwen (not Gemini) with a different prompt.
- Drawn from articles the 50-question set does not use.

Run the suite on it with `QA_SET=eval/qa_holdout.json`.

### Results (agent retrieval, k=5)

| Set | Index | R@1 | R@3 | R@5 | MRR | Misses |
|---|---|---|---|---|---|---|
| qa_set (50) | before | 76% | 94% | 96% | 0.848 | q001, q040 |
| qa_set (50) | **doc2query** | **82%** | 94% | 96% | **0.882** | q001, q040 |
| holdout (30) | before | 80% | 93% | 93% | 0.861 | h001, h014 |
| holdout (30) | **doc2query** | **83%** | **97%** | **97%** | **0.894** | h001 |

- **On the 50:** the labelled article moved up for q003, q035 (4th to 1st), q038, q048 and q027. It moved down for q012, q013 and q019, all still in the top 4.
- **q040 and q001 still miss.** q040 rose from 11th to 6th. For q001, the top hits are now redemption articles that answer the question ("book a ticket but not enough Enrich Points"); the label is the weak part.

**Alternative rejected: one vector per generated question.** This was tested in memory, scoring each article by its best vector. It was the only variant to reach R@5 100% on the 50, but on the held-out set R@1 fell to 63% and MRR to 0.758. Matching the LLM's own phrasing is the leakage this held-out set was built to expose.

### End to end (Gemini off, run 4 vs run 3)

38/50 cited the labelled article in both runs, so there was no net gain.

| Change | Questions |
|---|---|
| Now cite the labelled article (better ranking) | q025, q038 |
| Answered to clarify; labelled article still 1st, but a different 5th article made the grader ask for cabin or tier | q042, q043 |
| Clarify to declined (wrong either way) | q004, q036 |

Run 4 rows: `benchmarks/2026-10-16-gemini-thinking/runs/gemini-off_run4.json`. The test suite passes 40/40 on Qwen.

### Takeaway

Retrieval was not the binding constraint: 48/50 labelled articles were already retrieved, but only 38 were answered. Doc2query is kept because it improves ranking on both sets at no runtime cost. The remaining accuracy is in the grader's answer-or-clarify judgement.
