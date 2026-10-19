# Gemini 2.5 Flash thinking off vs dynamic, with local Qwen3-8B as reference

**AskMH RAG agent, end-to-end benchmark on 50 questions.** Runs recorded 2026-10-16.

| | |
|---|---|
| System | AskMH: LangGraph RAG agent answering only from the Malaysia Airlines AskMH FAQ (1,483 articles) |
| Configurations | `gemini-off` (Gemini 2.5 Flash, thinking budget 0), `gemini-dynamic` (thinking budget -1), `qwen` (Qwen3-8B on Ollama, reasoning off) |
| Dataset | `eval/qa_set.json`: 50 paraphrased questions, one labelled source article each, 42 topics |
| Runs | 2 per configuration (300 agent turns in total) |
| Script | `eval/compare_llms.py` |
| Raw data | `runs/*.json` next to this file (reply text removed) |

---

## 1. Summary

1. **Thinking did not improve accuracy.** Gemini with thinking off and with dynamic thinking both cited the
   correct article on 36/50 questions in run 1. In run 2 the figures were 36 and 37. The settings disagree on 8
   questions: thinking gained 3 correct answers and lost 2. Three of the 8 differences come from the
   *router*, not from the step that decides whether the evidence suffices.
2. **Thinking roughly doubled latency and multiplied output tokens by 11.** Median turn: 3.9 s with
   thinking off and 6.6 s with dynamic thinking. Mean, excluding the q004 outlier: 3.6 s and 9.7 s. Thinking-on runs had a long tail
   (54 s, 98 s turns) that never appears with thinking off. Output tokens per question, excluding one
   outlier: 76 off, 848 dynamic, of which 766 were thinking tokens.
3. **Gemini resolves far more questions than local Qwen3-8B.** Correct cited answers: Qwen 22/50,
   Gemini-off 36/50. 19 of the 26 differences between Qwen and Gemini-off arise at the grading step.
   Qwen asks for clarification much more often (16 vs 7). Qwen is faster per turn on a laptop GPU (median
   2.4–3.2 s) and fully deterministic across runs.
4. **Gemini's router degenerates on a curly apostrophe (a production bug).** For q004 ("MH Traveller**’**s
   Trio"), in every Gemini run, the router's JSON output stops at the `’` and fills the rest with tabs or
   newlines. That cost 21k–66k output tokens and 88–259 s per question, and retrieval then ran on a
   truncated query. This single question used 85% of thinking-off output tokens in run 2. Qwen handled it.
5. **Run-to-run stability.** Qwen gave identical results in both runs: same decisions, rewritten questions and
   token counts. Gemini thinking off was identical between runs on every outcome. Gemini dynamic thinking
   changed 1/50 outcomes.

**Recommendation for the hosted demo:** Gemini 2.5 Flash with thinking **off**. It is as accurate,
about 1.7x faster at the median, about 2.7x faster at the mean, and about 5x cheaper per question.
Separately, fix the router's degenerate output (Section 8.1).

---

## 2. Research questions

| RQ | Question |
|---|---|
| RQ1 | Does enabling Gemini 2.5 Flash's dynamic thinking change the agent's decisions (answer / clarify / decline / hand off) and its accuracy? |
| RQ2 | What does thinking cost in latency, both median and tail? |
| RQ3 | What does thinking cost in tokens and money? |
| RQ4 | How does a hosted model (Gemini 2.5 Flash) compare with the locally hosted Qwen3-8B the agent was developed on? |
| RQ5 | How stable are these results across repeated runs at temperature 0? |

---

## 3. System under test

The agent is a LangGraph state machine (`agent/graph.py`, `agent/nodes.py`, prompts in `agent/prompts.py`).
Each user turn runs:

```
router ──► retrieve ──► grade ──► generate        (answer with cited article URLs)
  │                       ├────► decline         (nothing relevant; counts a failure)
  │                       └────► (clarify)       (grader's clarifying question is the reply)
  ├──► greet / handoff (human requested) / decline (out of scope)
```

| Step | LLM call | What it does |
|---|---|---|
| `router` | Yes, structured output (JSON schema): `category` ∈ {greeting, human, faq, out_of_scope} and `question` | Classifies the message and rewrites it as a standalone question; the rewrite is the retrieval query |
| `retrieve` | No | Dense retrieval: `BAAI/bge-base-en-v1.5` embeddings, Chroma (cosine), top-k=5 *articles* (chunks over-fetched, then deduplicated per article URL) |
| `grade` | Yes, structured output: `decision` ∈ {answer, clarify, none}, `relevant` article numbers, `clarifying_question` | Decides whether the retrieved articles answer the question, or whether route, cabin or fare details are missing |
| `generate` | Yes, free text | Answers only from the graded articles. It may emit a fixed refusal line, which is converted into a decline with no sources |

A FAQ turn therefore makes 3 LLM calls (router, grade, generate), or 2 when the grader declines or asks to
clarify. Totals across 50 questions: 132 (Gemini-off run 2), 136 (Gemini-dynamic run 2), 121 (Qwen).

**Outcome labels** (`app/metrics.py: outcome_of`):

| Outcome | Meaning |
|---|---|
| `answered` | Grader said answer and the generator answered with sources |
| `clarify` | Grader asked a clarifying question |
| `declined` | Router said out of scope, the grader found nothing relevant, or the generator refused |
| `handoff` | Router classified the message as a request for a human |

---

## 4. Configurations

| | `gemini-off` | `gemini-dynamic` | `qwen` |
|---|---|---|---|
| Model | `gemini-2.5-flash` (Gemini API, paid tier) | `gemini-2.5-flash` | `qwen3:8b` (Ollama 0.35.1, local) |
| Thinking | `thinking_budget=0` | `thinking_budget=-1` (model decides) | `reasoning=False` |
| Temperature | 0 | 0 | 0 |
| Seed | n/a | n/a | 42 |
| Client rate limit | 60 req/min (LangChain `InMemoryRateLimiter`) | 60 req/min | none |
| Retries | SDK retries off (`max_retries=1`); app-level up to 4 attempts, 20 s wait on 429 | same | n/a |
| Where it runs | Google API, called from a laptop in Malaysia | same | Local GPU: RTX 4060 Laptop, 8 GB |

All three share the same retrieval: embeddings on the local GPU, the same Chroma index of 1,526 chunks, and k=5.

**Software:** `langchain` 1.4.2, `langgraph` 1.2.12, `langchain-google-genai` 4.4.0, `google-genai` 2.25.0,
`langchain-ollama` 1.1.0, `chromadb` 1.5.9, `sentence-transformers` 6.1.0, `torch` 2.11.0+cu128,
Python 3.11, Windows 11.

---

## 5. Dataset

`eval/qa_set.json`, built by `eval/build_qa_set.py`:

- 50 questions, stratified over 42 topics. Each was produced by an LLM paraphrasing an article and is
  labelled with that article's URL (`expected_url`).
- The questions deliberately avoid the article title's wording, so lexical overlap does not inflate retrieval.
- **Single-label caveat:** each question has exactly one "correct" article. The FAQ has near-duplicates; for
  example q038's label and a second article both describe preparing for a Live Chat session. An answer
  citing a near-duplicate scores as wrong here. Section 8.3 gives an example.

---

## 6. Method

**Procedure.** For each configuration and each run, every question is sent as the first message of a new
conversation, so there is no memory carry-over. The script streams graph updates (`stream_mode="updates"`) to time each step, and then reads the
final state. LangSmith tracing was disabled for the runs.

**Runs and ordering:**

| Run | What ran | Concurrency |
|---|---|---|
| Gemini run 1 | `gemini-off` and `gemini-dynamic` | In parallel (two processes) |
| Gemini run 2 | `gemini-off` and `gemini-dynamic` | In parallel (two processes) |
| Qwen run 1, then run 2 | `qwen` | Sequential, nothing else using the GPU except an idle API server |

Gemini run 1 used an earlier version of the script, which recorded no retrieved/cited URLs and no per-step
timing. Its outcomes, latency and tokens are comparable; retrieval columns show "n/a".

**Metrics:**

| Metric | Definition |
|---|---|
| answered / clarify / declined / handoff | Count of outcomes (Section 3) |
| **cited_expected** (accuracy) | Outcome is `answered` **and** the labelled article is among the cited sources |
| retrieval hit | Labelled article is among the 5 retrieved articles for the router's rewritten question |
| latency | Wall time of the whole turn. **The first question of each run is excluded** because it pays the embedding-model load (about 18 s), plus Qwen's model load |
| per-step latency | Time between consecutive streamed graph updates, i.e. the duration of each step |
| tokens | Sum of `usage_metadata` over all LLM calls in the turn (LangChain `get_usage_metadata_callback`). Gemini's `output_tokens` **includes** thinking tokens; `thinking_tokens` is `output_token_details.reasoning` |
| cost | Tokens × list price. Assumed Gemini 2.5 Flash paid tier: **$0.30 / 1M input, $2.50 / 1M output (thinking billed as output)**, as reported in May 2026; verify against the [official pricing page](https://ai.google.dev/gemini-api/docs/pricing) |

---

## 7. Results

### 7.1 Outcomes and accuracy (RQ1, RQ4)

| Run | answered | **cited_expected** | clarify | declined | handoff | error | retrieval hit |
|---|---|---|---|---|---|---|---|
| gemini-off r1 | 36 | **36** | 7 | 6 | 1 | 0 | n/a |
| gemini-off r2 | 36 | **36** | 7 | 6 | 1 | 0 | 43 |
| gemini-dynamic r1 | 38 | **36** | 8 | 3 | 1 | 0 | n/a |
| gemini-dynamic r2 | 39 | **37** | 7 | 3 | 1 | 0 | 44 |
| qwen r1 | 22 | **22** | 16 | 12 | 0 | 0 | 45 |
| qwen r2 | 22 | **22** | 16 | 12 | 0 | 0 | 45 |

- Every Gemini-off and Qwen answer cited the labelled article. Dynamic thinking produced 2 answers per run
  that cited a different article: q004 and q038 (Section 8).
- Agreement: **20** questions were answered correctly by all three configurations. **10** were never
  answered correctly by any run of any configuration: q001, q004, q005, q035, q036, q038, q040, q046, q047, q049.
- Retrieval hit counts (43–45 of 50) treat questions the router never sent to retrieval as misses.
  Counting only questions that reached retrieval, which runs on the *router's rewrite*:

  | Config (run) | Reached retrieval | Labelled article in top 5 | Real misses |
  |---|---|---|---|
  | gemini-off (r2) | 46 | 43 | q001, q004 (degenerate rewrite), q040 |
  | qwen (r1) | 47 | 45 | q001, q040 |

  This matches the 48/50 Recall@5 measured in Phase 5 on the raw questions, where q001 and q040 were
  also among the misses. Rank of the labelled article, Gemini-off r2: 1st = 35, 2nd = 6, 3rd = 1, 4th = 1.
  Qwen: 1st = 35, 2nd = 7, 3rd = 2, 4th = 1.
- Router classification, run 2: Gemini-off faq 46 / out_of_scope 3 / human 1; Gemini-dynamic 47 / 2 / 1;
  Qwen 47 / 3 / 0. Every question in the set is a genuine FAQ question, so every non-`faq` route is a
  router error.
  - **All** configurations routed q005 and q047 (car-rental refunds and fees, from the car-rental partner
    topic) `out_of_scope`. The router evidently does not consider car rental part of an airline FAQ.
  - The remaining misroutes: q019 (Gemini-off), q049 (Qwen), and Live Chat questions sent to `human`
    (q027 dynamic, q038 off).

### 7.2 Latency (RQ2)

Whole-turn wall time in seconds, first question excluded (n = 49):

| Run | mean | median | p90 | max |
|---|---|---|---|---|
| gemini-off r1 | 6.33 | 3.82 | 4.39 | 136.22 (q004) |
| gemini-off r2 | 5.31 | 3.91 | 4.22 | 88.66 (q004) |
| gemini-dynamic r1 | 11.91 | 6.67 | 9.10 | 254.93 (q004) |
| gemini-dynamic r2 | 14.75 | 6.56 | 9.14 | 259.34 (q004) |
| qwen r1 | 2.70 | 2.42 | 4.23 | 6.52 |
| qwen r2 | 3.42 | 3.21 | 4.75 | 8.84 |

Excluding the degenerate q004 as well (n = 48): Gemini-off mean 3.57 s, max 4.32 s; Gemini-dynamic mean
9.65 s, max 98.3 s; Qwen mean 2.72 s, max 6.52 s.

**Median duration of each step** (run 2 for Gemini, run 1 for Qwen):

| Step | gemini-off | gemini-dynamic | qwen |
|---|---|---|---|
| router | 1.32 s | 1.55 s | 0.80 s |
| retrieve | 0.05 s | 0.05 s | 0.01 s |
| grade | 1.33 s | **2.63 s** | 0.87 s |
| generate | 1.30 s | **2.09 s** | 1.44 s |

Thinking adds the most time at the grading step (about 2x), where the model weighs evidence. The router,
a simple classification, barely changes.

**Tail latency, dynamic thinking:** apart from q004, run 2 had turns of 98.3 s (q009, generate step 94.0 s),
55.7 s (q040, grade step 54.1 s) and 25.9 s (q036, grade step 24.1 s). With thinking off, no turn other
than q004 exceeded 4.4 s. Dynamic thinking makes latency unpredictable; the median hides that.

The Qwen run-1/run-2 latency difference (median 2.42 s vs 3.21 s) reflects GPU and thermal state on a laptop.
The work done was identical (same tokens and outputs).

### 7.3 Tokens and cost (RQ3)

Totals across 50 questions:

| Run | input tokens | output tokens | of which thinking | cost (list price) |
|---|---|---|---|---|
| gemini-off r2 | 53,309 | 25,259 | 0 | $0.079 |
| gemini-dynamic r2 | 170,268 | 107,487 | 37,940 | $0.320 |
| qwen r1 | 51,525 | 4,514 | 0 | local (no API cost) |

These totals are dominated by the q004 outlier. **Excluding q004 (n = 49):**

| Config | input tokens | output tokens | of which thinking | output / question | cost / question |
|---|---|---|---|---|---|
| gemini-off | 52,185 | 3,719 | 0 | 76 | **$0.0005** |
| gemini-dynamic | 54,260 | 41,568 | 37,533 | 848 | **$0.0024** |
| qwen | 50,597 | 4,442 | 0 | 91 | n/a |

- Dynamic thinking multiplies output tokens by about **11x**, and thinking tokens are 90% of its output. Cost per
  question goes up about **5x**. Input is nearly unchanged.
- The q004 failure alone cost about **$0.054** (thinking off, run 2: 21,540 output tokens), **$0.090** (off,
  run 1: 35,880) and **$0.200** (dynamic, run 2: 65,919 output and 116,008 input tokens). The cause of the
  inflated input in the dynamic run was not determined. The installed `google-genai` 2.25.0 has no automatic
  continuation, so it is not an SDK re-send. One such question costs as much as 100–400 normal ones.

### 7.4 Stability across runs (RQ5)

| Config | Outcomes changed, r1 → r2 | Other differences |
|---|---|---|
| qwen | 0 / 50 | Rewritten questions identical 50/50; token totals identical |
| gemini-off | 0 / 50 | Output tokens differed by 1–9 tokens on 9 questions (wording of answers or clarifying questions), and widely on q004 (degenerate output length varies) |
| gemini-dynamic | 1 / 50 (q033: clarify → answered ✓) | |

At temperature 0, Gemini without thinking was as repeatable as seeded Qwen on these 50 questions. With
thinking it was slightly less so. Two runs are too few to estimate variance properly (Section 9).

---

## 8. Analysis of the differences

To find where two configurations diverge, I checked the pipeline in order and attributed each difference to the first step that disagrees: router (`route`), then retrieval (`retrieved` set), then grader (`decision`), then generator.

| Comparison | Questions that differ | Router | Retrieval | Grader | Generator | Correct answers gained / lost by the second config |
|---|---|---|---|---|---|---|
| gemini-off → gemini-dynamic (r2) | 8 | 3 | 1 | 4 | 0 | +3 / −2 |
| qwen → gemini-off | 26 | 3 | 2 | 19 | 2 | +16 / −2 |
| qwen → gemini-dynamic | 24 | 2 | 2 | 18 | 2 | +16 / −1 |

### 8.1 Thinking off vs dynamic, question by question (run 2)

✓ = answered citing the labelled article.

| id | Question | Off | Dynamic | Where they diverge | Notes |
|---|---|---|---|---|---|
| q004 | Can I buy MH Traveller’s Trio for a domestic flight from KLIA? | clarify | answered (other article) | retrieval | Both routers produced a degenerate rewrite (below), so the retrieved sets differ. Dynamic cited a related Traveller's Trio article ("When and where can I purchase…"), not the labelled one ("Is MH Traveller's Trio applicable system-wide…") |
| q006 | How can I stop my card from being charged again for future flights? | declined | answered ✓ | grader | Same retrieval (labelled article at rank 1). Thinking let the grader see that the article answers the question |
| q019 | Can I bring a pet in my extra baggage? | declined | answered ✓ | **router** | Thinking off routed it `out_of_scope`; dynamic routed it `faq` |
| q027 | Can I use Live Chat if I'm not in Malaysia? | answered ✓ | handoff | **router** | Dynamic read "Live Chat" as a request for a human |
| q033 | What should I do if my flight is delayed and I miss my train? | clarify | answered ✓ | grader | Dynamic was itself unstable here: clarify in run 1, answered in run 2 |
| q038 | What documents should I bring to the Live Chat for my Malaysia Airlines flight? | handoff | answered (near-duplicate article) | **router** | Off read "Live Chat" as a request for a human. Dynamic cited "What information should I have ready before starting a Live Chat session", a near-duplicate of the label ("…before starting a session for FY Live Chat"). Arguably correct |
| q040 | How do I add a new card to my account? | declined | clarify | grader | The labelled article was not retrieved in either run, so both outcomes are wrong |
| q043 | Can I use the Golden Lounge for arriving flights? | answered ✓ | clarify | grader | Thinking made the grader *more* cautious here |

Net effect of thinking on the grader: +2 correct answers (q006, q033), −1 (q043), plus one wrong outcome
swapped for another (q040). The router-level flips (q019, q027, q038) go both ways and look like
noise in how the router reads ambiguous wording, not a systematic benefit.

**Router failure: degenerate structured output (q004, every Gemini run).** The question contains U+2019
(`’`). Gemini's router, which returns JSON under a response schema, produced:

```
"question": "Can I buy MH Traveller\n\t\t\t\t\t\t\t\t\t\t\t\t\t ... (tens of thousands of whitespace tokens)
```

| Run | Router step time | Output tokens (turn) |
|---|---|---|
| off r1 | ≈136 s turn | 35,880 |
| off r2 | 87.2 s | 21,540 |
| dynamic r1 | ≈255 s turn | 66,221 |
| dynamic r2 | 253.5 s | 65,919 |

Qwen rewrote the same question correctly (`Can I buy MH Traveller’s Trio …`) in 1.9 s.

- **Impact:** any user who types a curly apostrophe, as phone keyboards insert by default, risks a turn
  of minutes, a retrieval query cut off at the apostrophe, and a cost of $0.05–$0.20 instead of $0.0005.
- **Likely mechanism:** while escaping a non-ASCII character inside a schema-constrained JSON string, the
  model falls into a repetition loop. Neither temperature 0 nor turning thinking off prevents it.
- **Mitigations, implemented after this benchmark** (`agent/nodes.py`; see the addendum below):
  1. Normalise typographic punctuation (`’ ‘ “ ”`, dashes, ellipsis, non-breaking space) to ASCII before the router.
  2. Cap the router's output at 256 tokens. Gemini's router also runs without thinking, whose tokens would
     count against the cap.
  3. Fall back to the original user message when the rewrite looks degenerate (runs of tabs or newlines),
     and treat unparseable router output as an FAQ question instead of failing the turn.

**Addendum: fix verified 2026-10-18** (live Gemini 2.5 Flash, thinking off):

| Check | Before the fix | After the fix |
|---|---|---|
| q004 through the router | Degenerate rewrite, 21k–66k output tokens, 88–254 s | `route=faq`, rewrite `Can I buy MH Traveller's Trio for a domestic flight from KLIA?`, 1.2 s |
| Raw U+2019 message, bypassing normalisation (cap and fallback only) | (same as above) | Stopped at 243 output tokens in 2.3 s; the degenerate rewrite was rejected and the original message used |
| Router output cap | none | 256 tokens, `finish_reason=MAX_TOKENS` on a runaway prompt; the main model is unaffected |
| Test suite | 29 tests | 38 pass on Qwen; `tests/test_3_router.py` (18) passes on Gemini |

Router answers in Sections 7 and 8 were recorded before the fix, so q004's outcome would differ in a re-run.

**Addendum 2: router and grader prompt fix, 2026-10-19** (run 3, `runs/*_run3.json`)

Most declines in Section 7.1 came from the router and grader, not retrieval. Two prompt changes in `agent/prompts.py`:

1. **Router:** partner and add-on services (car rental, hotels, MHrail, cargo and pets, gift cards, bundles) are now `faq`; `out_of_scope` means nothing to do with travel or Malaysia Airlines. Questions *about* Live Chat are `faq`; only wanting to talk to someone now is `human`. This addresses follow-up 2 in Section 10.
2. **Grader:** the `answer` rule adds "Its wording does not have to match the question's." A stronger rewrite (answer "fully or in part"; `none` only when every article is on a different subject) was tried first and rejected: on Qwen it answered the ambiguous "how much is extra baggage?" instead of asking for the route, and accepted baggage articles for a car-rental-in-Paris question. Both are covered by tests that failed.

| Config | Run | answered | cited_expected | clarify | declined | handoff | median turn |
|---|---|---|---|---|---|---|---|
| gemini-off | r2 (before) | 36 | 36 | 7 | 6 | 1 | 3.91 s |
| gemini-off | **r3 (after)** | **40** | **38** | 7 | **3** | **0** | 3.82 s |
| qwen | r1 (before) | 22 | 22 | 16 | 12 | 0 | 2.42 s |
| qwen | **r3 (after)** | **30** | **30** | 17 | **3** | 0 | 2.77 s |

- **Gemini, router:** q005, q047 (car rental) and q019 (pet in extra baggage) moved from `out_of_scope` to answered with the labelled article. q038 is no longer handed off and now cites the near-duplicate Live Chat article.
- **Gemini, other changes:**
  - q036 (36 weeks pregnant) moved from declined to clarify.
  - q025 now cites two MHflypass articles that say the same thing as the label, so it no longer counts as `cited_expected`.
  - q001 moved from clarify to declined; its labelled article is not retrieved in either run.
- **Qwen, router:** car rental and pet cargo questions (q005, q047, q049) now reach retrieval.
- **Qwen, grader:** q003, q007, q008, q020, q023, q036, q042 and q044 moved to answered with the labelled article. q018 regressed from answered to declined.
- **Qwen baseline caveat:** run 1 predates the 2026-10-18 router hardening, so q004 (declined to clarify) reflects that fix, not this one.
- **Remaining declines:**
  - Gemini: q001 and q040 (retrieval misses) and q006 (grader).
  - Qwen: q018 (grader), plus q026 and q048, where its generator still refuses after the grader said answer.
- Test suite: 40/40 pass on Qwen with the new prompts.

### 8.2 Gemini vs Qwen

19 of the 26 differences between Qwen and Gemini-off are at the grader. With the same retrieved articles,
Qwen3-8B is far more conservative: in 16 of those cases Gemini answered correctly where Qwen asked for
clarification (e.g. q007, q008, q013, q034, q042, q044) or declined (e.g. q016, q020, q043).

Qwen also shows a failure Gemini never did: twice (q026, q048) its grader said *answer* and its own
generator then refused, producing a decline.

Router-level: Qwen routed q049 ("How much does it cost to send my pet by air?") `out_of_scope`; Gemini-off
did the same with q019.

The reverse direction is small: Qwen answered correctly on only 2 questions where Gemini-off did not
(q030, q033). In both, Gemini-off asked for clarification.

### 8.3 Questions no configuration answered correctly

q001, q004, q005, q035, q036, q038, q040, q046, q047, q049. Causes seen in the data:

| Cause | Questions |
|---|---|
| Router: car rental judged out of scope by every configuration | q005, q047 |
| Retrieval miss: the labelled article is not in the top 5 for the rewritten question | q001, q040 (and q004, via the degenerate Gemini rewrite) |
| Grader caution: the labelled article was retrieved, but the outcome was clarify or decline | q035 (rank 4), q036 (rank 1), q046 (rank 1), q049 (rank 1, Gemini; Qwen routed it out of scope) |
| Router: Live Chat questions read as a request for a human, plus label ambiguity (near-duplicate articles) | q038 |

---

## 9. Threats to validity

1. **Small sample.** 50 questions and 2 runs per configuration. One question is 2 percentage points; the
   36 vs 37 difference in accuracy is within noise. No confidence intervals were computed. Recommended: ≥5
   runs per configuration and a larger question set.
2. **Single-label scoring.** Near-duplicate FAQ articles (q038) and multiple valid sources (q004) make
   `cited_expected` a lower bound on true answer quality. Answer *content* was not graded (no
   faithfulness or LLM-as-judge check), so an answer citing the right article is assumed correct.
3. **Clarify is not always wrong.** Some questions really depend on fare or route. This benchmark counts
   every clarify as unresolved.
4. **Synthetic questions.** LLM paraphrases of article content, not real user traffic. Real users write
   shorter, noisier queries, including curly quotes (Section 8.1).
5. **Latency conditions.**
   - Gemini off and dynamic ran concurrently in both runs. They shared network and client CPU, and the
     embedding model ran per process on one GPU.
   - Gemini latency includes the network round-trip from Malaysia to Google's global endpoint.
   - Qwen latency depends on laptop GPU and thermal state.
   - The client rate limiter (60 req/min) can add up to ~1 s between calls; most calls take longer than that.
6. **Pricing assumption.** Costs use list prices reported in May 2026 ($0.30 / $2.50 per 1M). Check the
   official page before citing.
7. **Gemini run 1 is partial.** It recorded no retrieval or per-step data, but its outcomes, latency and token
   totals are comparable.
8. **Model versioning.** `gemini-2.5-flash` is a hosted alias that may be updated server-side. Re-running later
   may not reproduce these numbers exactly.

---

## 10. Conclusions

- **RQ1:** Dynamic thinking does not measurably improve the agent's accuracy on this set (36 vs 36–37 correct).
  It changes about 16% of outcomes in both directions, and a third of those changes come from the router.
- **RQ2:** Thinking roughly doubles median latency at the grading step and introduces a heavy tail (25–98 s
  turns) that thinking off does not have.
- **RQ3:** Thinking costs about 11x the output tokens and about 5x the money per question.
- **RQ4:** Gemini 2.5 Flash resolves 64% more questions than Qwen3-8B with the same retrieval and prompts. The gap
  is almost entirely the grader's willingness to commit to an answer. Qwen is faster on local hardware,
  fully deterministic and free, but resolves only 44% of questions.
- **RQ5:** At temperature 0, Qwen (seeded) and Gemini without thinking were fully repeatable over two runs;
  Gemini with thinking changed 1/50 outcomes.
- **Unplanned finding:** a schema-constrained structured-output failure on U+2019 is the largest single cost
  and latency risk in the hosted system, larger than the choice of thinking mode.

**Decision:** deploy with `GEMINI_THINKING_BUDGET=0`. Follow-ups:
1. Router hardening (Section 8.1 mitigations).
2. Clarify the router prompt so FAQ questions *about* Live Chat are not treated as requests for a human,
   and so that partner services such as car rental count as in scope.
3. Examine grader caution on q035, q036 and q046.
4. Re-run with ≥5 repetitions.

---

## 11. Reproduction

```powershell
conda activate askmh
$env:PYTHONPATH = (Get-Location).Path
# needs GOOGLE_API_KEY (paid tier) in .env and Ollama serving qwen3:8b
python -m eval.compare_llms --gemini --run 2            # both Gemini configs in parallel
python -m eval.compare_llms --config qwen --run 1       # one config
python -m eval.compare_llms --report                    # summary.json over all saved runs
python -m eval.compare_llms --export <dir>              # rows without reply text
```

Raw runs, including full replies, are written to `eval/results/bench/` (gitignored). The files in `runs/` are
the same rows without the `reply` field, because replies quote Malaysia Airlines content.

**Row schema** (`runs/<config>_run<n>.json` → `rows[]`):
- question: `id`, `question`, `topic`, `expected_url`
- routing and retrieval: `route`, `rewritten_question`, `retrieved` (top-5 article URLs), `expected_rank`
- grading and outcome: `graded_relevant`, `decision`, `outcome`, `sources`, `cited_expected`
- timing: `wall_s`, `nodes` (`[step, seconds]`)
- usage: `llm_calls`, `tokens_in`, `tokens_out`, `thinking_tokens`

`meta` records the configuration, model, thinking budget, rate limit, k and timestamps.

---

## Appendix A: per-question results

Outcome codes: **A** answered, **C** clarify, **D** declined, **H** handoff; **✓** = cited the labelled article.

- **rank Off·Dyn·Qwen:** position of the labelled article among the 5 retrieved, from Gemini-off run 2,
  Gemini-dynamic run 2 and Qwen run 1 (– = not retrieved).
- **s Off r2, s Dyn r2, s Qwen r1:** turn wall time in seconds for those runs. The first question includes
  model loading.

| id | topic | question | rank Off·Dyn·Qwen | Off r1 | Off r2 | Dyn r1 | Dyn r2 | Qwen r1 | Qwen r2 | s Off r2 | s Dyn r2 | s Qwen r1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q001 | AmBank Enrich Visa Credit Cards | How do I use my Enrich Points to book a flight? | –·–·– | C | C | C | C | C | C | 22.5 | 38.8 | 20.2 |
| q002 | Baggage | How is the extra baggage fee determined based on flight distance? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.3 | 6.7 | 3.7 |
| q003 | Becoming An Enrich Member | Can I use the same Enrich account for both work and personal travel? | 2·2·2 | A✓ | A✓ | A✓ | A✓ | D | D | 3.9 | 6.7 | 2.4 |
| q004 | Bundle Products | Can I buy MH Traveller’s Trio for a domestic flight from KLIA? | –·–·1 | C | C | A | A | D | D | 88.7 | 259.3 | 1.9 |
| q005 | Car Rental | Will my car rental refund depend on the rental company's terms and conditions? | –·–·– | D | D | D | D | D | D | 1.4 | 1.3 | 0.7 |
| q006 | Card Payment Fee | How can I stop my card from being charged again for future flights? | 1·1·1 | D | D | A✓ | A✓ | C | C | 2.6 | 7.0 | 1.9 |
| q007 | Cargo Services | What size should my animal container be for the flight? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | C | C | 3.7 | 5.5 | 1.7 |
| q008 | Check-in | Can I use a mobile boarding pass for my flight from Kuala Lumpur to another city? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | C | C | 3.8 | 5.8 | 2.1 |
| q009 | Check-in Services | Can I get help with check-in and boarding if I'm a special needs passenger? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.3 | 98.3 | 3.5 |
| q010 | Chef-on-Call | Can I bring my own meal on the flight? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.2 | 7.7 | 2.5 |
| q011 | Customer Relations | What happens if I send feedback to customer@malaysiaairlines.com after March 14, 2020? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.1 | 7.5 | 5.0 |
| q012 | Disability Assistance | Do I get a discount with my OKU card for a Malaysia Airlines flight? | 2·2·2 | A✓ | A✓ | A✓ | A✓ | C | C | 3.7 | 5.5 | 1.9 |
| q013 | Disruption Letter, Confirmation Letter | Can I get a copy of my boarding pass through the Malaysia Airlines website? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | C | C | 4.1 | 6.8 | 2.2 |
| q014 | Earning Enrich Points | Will I still get Enrich Points if my Malaysia Airlines flight is cancelled and I take another airline? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.9 | 5.2 | 2.4 |
| q015 | Economy Pre-Order Meals | Will my ordered economy meal be available if there's a flight delay? | 2·2·2 | A✓ | A✓ | A✓ | A✓ | C | C | 3.9 | 6.0 | 1.8 |
| q016 | Enrich | Will I get my OTP on my international number if I'm using a GSM network? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | D | D | 3.9 | 7.1 | 1.7 |
| q017 | Enrich Benefits And Privileges | Do my Enrich Platinum Points expire if I get downgraded to Gold? | 2·2·2 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.2 | 6.6 | 2.5 |
| q018 | Enrich Redemption | Will I get Enrich Points for the cash part of my Travel Award booking? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.9 | 8.2 | 2.4 |
| q019 | Extra Baggage | Can I bring a pet in my extra baggage? | –·2·2 | D | D | A✓ | A✓ | C | C | 1.3 | 7.7 | 2.0 |
| q020 | Extra Legroom And 10kg Extra Baggage | Can I get an extra legroom seat and 10kg baggage allowance together? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | D | D | 4.0 | 6.1 | 2.1 |
| q021 | Fare Freeze | Can I get a refund for the Fare Freeze holding fee if my flight was canceled? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.9 | 5.9 | 2.2 |
| q022 | Fit to Travel | Can I bring my medication in my carry-on bag? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.2 | 5.8 | 2.1 |
| q023 | Fix Your Plan | Will my extra baggage fee still apply if I get a new booking after using Fix Your Plan? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | C | C | 4.0 | 7.1 | 2.3 |
| q024 | Golden Lounge | Where can I find Malaysia Airlines lounge partners besides those at KLIA? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.8 | 7.0 | 3.0 |
| q025 | Ground Products & Services | Can I transfer or refund my MHflypass ASEAN membership? | 2·2·2 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.7 | 5.9 | 2.7 |
| q026 | Inflight Products & Services | Can I take the meal from the plane and eat it later? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | D | D | 4.0 | 6.9 | 1.6 |
| q027 | Live Chat | Can I use Live Chat if I'm not in Malaysia? | 3·–·3 | A✓ | A✓ | H | H | D | D | 3.9 | 1.6 | 2.3 |
| q028 | MH Premium Transfer | Can I use the transfer service without booking ahead on my departure day? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.0 | 5.9 | 3.2 |
| q029 | MHcheckout | Will my credit card information stay private when booking with MHcheckout? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.9 | 5.5 | 2.7 |
| q030 | MHexplorer | How do I get help if I can't log in or receive my OTP? | 1·1·1 | C | C | C | C | A✓ | A✓ | 2.9 | 5.5 | 3.7 |
| q031 | MHflypass ASEAN | How do I redeem my MHflypass ASEAN for a flight? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.8 | 7.5 | 5.8 |
| q032 | MHgiftcard | Can I add more money to my MHgiftcard after I buy it? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.7 | 5.1 | 2.6 |
| q033 | MHrail | What should I do if my flight is delayed and I miss my train? | 1·1·1 | C | C | C | A✓ | A✓ | A✓ | 2.8 | 6.7 | 2.7 |
| q034 | MHsports | Can multiple passengers in my MHsports group share one baggage allowance? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | C | C | 4.1 | 6.6 | 2.2 |
| q035 | Manage My Booking | Do I need to pay extra if I want to change my flight date? | 4·4·4 | C | C | C | C | C | C | 2.9 | 5.8 | 2.1 |
| q036 | Medical Assistance | Can I travel on a Malaysia Airlines flight if I'm 36 weeks pregnant? | 1·1·1 | D | D | D | D | C | C | 2.6 | 25.9 | 2.6 |
| q037 | Online Booking | Can I change my name on my Malaysia Airlines ticket after a marriage or divorce? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.1 | 7.0 | 4.9 |
| q038 | Others | What documents should I bring to the Live Chat for my Malaysia Airlines flight? | –·3·3 | H | H | A | A | C | C | 1.3 | 5.9 | 3.5 |
| q039 | Refunds | How do I get a refund for my ticket after a long flight delay? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.1 | 6.3 | 4.2 |
| q040 | Reservation & Ticketing | How do I add a new card to my account? | –·–·– | D | D | C | C | C | C | 2.6 | 55.7 | 2.6 |
| q041 | Senior Citizens | Where can I get help if I'm an elderly passenger flying alone? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.9 | 9.1 | 4.7 |
| q042 | Special Assistance | Can my child with special needs travel alone on the flight? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | C | C | 4.3 | 6.6 | 2.8 |
| q043 | AmBank Enrich Visa Credit Cards | Can I use the Golden Lounge for arriving flights? | 1·1·1 | A✓ | A✓ | C | C | D | D | 4.2 | 5.2 | 2.9 |
| q044 | Baggage | Can I check 2kg of cooking oil in my luggage for a flight? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | C | C | 3.9 | 5.9 | 2.4 |
| q045 | Becoming An Enrich Member | Does my Enrich Blue membership expire if I don't use it? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 3.8 | 5.4 | 2.7 |
| q046 | Bundle Products | Does the extra legroom and baggage stay valid if the flight is delayed? | 1·1·1 | C | C | C | C | D | D | 2.8 | 13.3 | 2.6 |
| q047 | Car Rental | Are the local taxes and road fees included in the rental price? | –·–·– | D | D | D | D | D | D | 1.5 | 1.5 | 1.0 |
| q048 | Card Payment Fee | Why did my card show two charges for one flight booking? | 2·2·2 | A✓ | A✓ | A✓ | A✓ | D | D | 4.3 | 6.1 | 2.2 |
| q049 | Cargo Services | How much does it cost to send my pet by air? | 1·1·– | C | C | C | C | D | D | 2.8 | 8.4 | 1.0 |
| q050 | Check-in | Why do I need to provide my passport and visa info when checking in for my flight? | 1·1·1 | A✓ | A✓ | A✓ | A✓ | A✓ | A✓ | 4.2 | 6.8 | 6.5 |
