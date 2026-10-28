# AskMH Chatbot

AskMH is an agentic RAG support assistant that answers Malaysia Airlines passenger questions using only a 1,483-article knowledge base. It cites its sources, asks for missing details, and hands off to a human when it can't help.
Under the hood, a LangGraph state machine routes each message, retrieves and grades relevant articles, and generates grounded answers through typed Pydantic decision points.
It is served by a FastAPI backend with Google sign-in and a React frontend, and is deployed on Google Cloud Run with LangSmith tracing and an admin analytics dashboard.

**Live demo:** https://askmh-194250607947.asia-southeast1.run.app (Google sign-in required, to limit llm model quota)

![AskMH answering a baggage question with a cited source](assets/chat.png)

Non-commercial portfolio project, not affiliated with Malaysia Airlines. The FAQ content belongs to Malaysia Airlines.

## Architecture

```mermaid
flowchart LR
    subgraph Offline["Offline: build the knowledge base"]
        A[AskMH sitemap] --> B[Playwright scraper]
        B --> C[Markdown articles<br/>1,483 articles, 41 topics]
        C --> D[Chunker<br/>whole article if ≤1,500 chars]
        D --> E[bge-base-en-v1.5<br/>embeddings]
        E --> F[(Chroma<br/>1,526 chunks)]
    end

    subgraph Online["Online: answer a question"]
        U[React UI] -->|/api/chat| API[FastAPI]
        API --> G[LangGraph agent]
        G -->|dense search| F
        G --> LLM[Gemini 2.5 Flash<br/>or local Qwen3-8B]
        API --> M[(SQLite metrics)]
        M -.->|every 10 min| HF[(HF Dataset backup)]
    end
```

### Agent flow

Each user message runs through a LangGraph state machine. A FAQ question takes three LLM calls: router, grader and generator.

```mermaid
flowchart TD
    S([User message]) --> R{Router}
    R -->|greeting| GR[Greet]
    R -->|asks for a human| H[Hand off to live chat]
    R -->|out of scope| D[Decline]
    R -->|FAQ question| RT[Retrieve top 5 articles]
    RT --> GD{Grader}
    GD -->|articles answer it| GEN[Generate answer<br/>with cited sources]
    GD -->|needs route, cabin or fare| C[Ask a clarifying question]
    GD -->|nothing relevant| D
    GEN -->|articles turn out not to answer| D
    D -->|2nd failure| H
    GR --> E([Reply])
    GEN --> E
    C --> E
    D --> E
    H --> E
```

- **Router** classifies the message and rewrites it as a standalone question, using the conversation so far. The rewrite becomes the search query.
- **Retrieve** runs a dense search in Chroma, over-fetches chunks and keeps the top 5 distinct articles.
- **Grader** decides whether those articles answer the question, need a clarifying question, or are irrelevant.
- **Generate** answers only from the graded articles and cites their URLs. If they don't actually answer the question, it refuses, and that counts as a failure.
- **Handoff** triggers when the user asks for a human or after 2 failures. Live chat hours are 7:00 am to 11:00 pm MYT.

### Components

| Layer | Tech | Folder |
|---|---|---|
| Ingestion | Playwright (the site is a JS-rendered Salesforce portal), BeautifulSoup, markdownify | `ingest/` |
| Indexing | Hugging Face `BAAI/bge-base-en-v1.5`, Chroma (cosine) | `index/` |
| Agent | LangChain, LangGraph, Gemini 2.5 Flash or Ollama `qwen3:8b` | `agent/` |
| Backend | FastAPI, Google sign-in, per-user and global quotas | `api/` |
| Frontend | React 19, TypeScript, Vite, Tailwind v4 | `web/` |
| Metrics | SQLite (outcomes, feedback, source clicks), mirrored to a private HF Dataset | `app/` |
| Evaluation | pytest layers, 50-question QA set, LLM benchmark | `tests/`, `eval/` |
| Hosting | One Docker container on Google Cloud Run, scales to zero | `Dockerfile`, `deploy/` |

### Design choices

- **Dense retrieval only.** Hybrid dense + BM25 scored lower than plain dense on the 50-question eval, so Hybrid was dropped.
- **Doc2query.** Gemini wrote 4 passenger-style questions per article, embedded with the article but never shown to the answering model. It lifted MRR from 0.848 to 0.882, and to 0.894 on held-out questions.
- **Small articles stay whole.** 98% of articles are under 1,500 characters, so they are embedded as one chunk. Longer ones split at paragraph breaks, and tables are never split.
- **Gemini thinking off.** It gave the same accuracy as dynamic thinking, at about half the median latency and a fifth of the cost.
- **Sign-in protects the quota, not the key.** The Gemini key never leaves the server. Google sign-in only gates who can spend the shared daily quota.
- **No stored conversations.** Chat memory is in-memory and deleted when the user refreshes, closes the tab or starts a new chat. Only anonymous metrics persist.

## Results

**Retrieval** (50 paraphrased questions, worded differently from the article titles): Recall@5 **96%** (48/50), MRR **0.882**. On a 30-question held-out set of short casual questions: Recall@5 **97%**, MRR **0.894**.

**End-to-end** (50 questions, full agent, recorded 2026-10-19):

| Model | Answered with a cited source | Asked a clarifying question | Declined | Median turn |
|---|---|---|---|---|
| Gemini 2.5 Flash, thinking off | 40 | 7 | 3 | 3.8 s |
| Qwen3-8B, local RTX 4060 | 30 | 17 | 3 | 2.8 s |

A clarifying question is a valid outcome, not a failure: many AskMH answers depend on the route, cabin or fare. Counting both, each model handles **47/50 (94%)** without declining. Every Qwen answer cited the article labelled in the test set. Gemini's did in 38 of 40, and the other 2 cited a near-duplicate article with the same content.

Before a prompt fix on 2026-10-19, Gemini declined 6 questions and Qwen 12. The router was rejecting partner services such as car rental and pet travel as out of scope, and the grader was declining articles that answered the question in different words.

**Tests:** 40 pass on local Qwen3-8B, across 7 layers tested bottom-up: retrieval, structured output, router, grader, generation, full graph and API sessions.

## Project structure

```
config.py     settings, read from .env
ingest/       sitemap, Playwright scraper, HTML to Markdown
index/        chunker, embeddings, index builder, search CLI
agent/        retriever, LangGraph nodes and graph, prompts, terminal chat
api/          FastAPI app, auth, quotas, schemas
app/          SQLite metrics and HF Dataset sync
web/          React frontend
tests/        layered test suite
eval/         QA set, harness, findings, benchmarks
deploy/       Cloud Run walkthrough
```

`data/` (scraped articles and the Chroma index) is gitignored, so you build it locally with steps 1 and 2 below.

## Run it locally

### Setup

```powershell
conda create -n askmh python=3.11 -y
conda activate askmh
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m playwright install chromium
copy .env.example .env
```

Every Python command below assumes `conda activate askmh` and the repo root as the working directory.

Ollama must be running with `qwen3:8b` pulled (`ollama pull qwen3:8b`). To use Gemini instead, set `CHAT_PROVIDER=gemini` and `GOOGLE_API_KEY` in `.env`.

### 1. Scrape AskMH

```powershell
python -m ingest.scraper            # incremental: only new/changed articles (by sitemap lastmod)
python -m ingest.scraper --limit 5  # quick test
python -m ingest.scraper --force    # re-scrape everything
```

Output: `data/articles/<topic>/<article-slug>.md`, one file per article with YAML frontmatter (`title`, `url`, `topic`, `last_updated`, ...). Tables with merged cells are expanded into full Markdown grids. A few sitemap entries redirect to a login page and are skipped as `LoginRequired` in `data/failed.json`.

### 2. Build the index

```powershell
python -m index.build_index                          # ~2 min on GPU
python -m index.search "power bank" --mode dense     # dense | bm25 | hybrid
```

### 3. Chat in the terminal

```powershell
python -m agent.chat --debug
```

### 4. Run the app

Terminal 1, the API (wait for `Application startup complete`):

```powershell
python -m uvicorn api.main:app --port 8001           # Swagger at http://127.0.0.1:8001/docs
```

Terminal 2, the React frontend (see [web/README.md](web/README.md)):

```powershell
cd web
npm install
npm run dev                                          # http://localhost:5173
```

API routes live under `/api` (`/api/health`, `/api/greeting`, `/api/chat`, `/api/feedback`, `/api/config`). `/go` stays at the root because source links open in a new tab without an auth header. If `web/dist` exists, the API also serves the built frontend at `/`, which is how the container runs everything on one port:

```powershell
cd web; npm run build; cd ..
python -m uvicorn api.main:app --port 8001          # now serves the UI at / too
```

### Sign-in (optional locally, required for the hosted demo)

Off by default, so nothing above needs credentials. To try it, set in `.env`:

```
AUTH_ENABLED=true
GOOGLE_OAUTH_CLIENT_ID=<your web client id>.apps.googleusercontent.com
SESSION_SECRET=<python -c "import secrets; print(secrets.token_urlsafe(32))">
```

The API refuses to start if sign-in is on and `SESSION_SECRET` is missing, too short or left at the default, since a weak secret lets anyone forge a session. Per-user and global daily caps (`DAILY_MESSAGES_PER_USER`, `DAILY_MESSAGES_GLOBAL`) protect the shared quota and return 429 when hit.

### 5. Tests

```powershell
python -m eval.run_suite                             # all layers
python -m pytest tests/test_3_router.py -v           # one layer
```

### Deploy

See [deploy/DEPLOY.md](deploy/DEPLOY.md) for the Cloud Run walkthrough.
