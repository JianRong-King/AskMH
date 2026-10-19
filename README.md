# AskMH Chatbot

RAG chatbot answering Malaysia Airlines passenger FAQs strictly from the [AskMH knowledge base](https://www.askmh.malaysiaairlines.com/faq/s/), citing the source article and handing off to (simulated) live chat when it can't answer.

Stack: Playwright, Hugging Face `bge-base-en-v1.5` embeddings, Chroma, LangChain + LangGraph, Ollama `qwen3:8b` (or Gemini), FastAPI, React + TypeScript + Tailwind.

**Live demo**: _(add the Space URL here once deployed)_ — sign-in with Google is required because the demo runs on a shared free-tier model quota. Deployment steps are in [deploy/DEPLOY.md](deploy/DEPLOY.md).

Engineering write-up of the defects the test suite found, and how they were fixed: [eval/FINDINGS.md](eval/FINDINGS.md).

Non-commercial portfolio project. The FAQ content belongs to Malaysia Airlines.

## Setup

```powershell
conda create -n askmh python=3.11 -y
conda activate askmh
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m playwright install chromium
copy .env.example .env
```

Every Python command below assumes `conda activate askmh` and the repo root as the working directory. Plain `python` outside the env is a different interpreter without the project's packages.

Ollama must be running with `qwen3:8b` pulled (`ollama pull qwen3:8b`). To use Gemini instead, set `CHAT_PROVIDER=gemini` and `GOOGLE_API_KEY` in `.env`.

## 1. Scrape AskMH

```powershell
python -m ingest.scraper            # incremental: only new/changed articles (by sitemap lastmod)
python -m ingest.scraper --limit 5  # quick test
python -m ingest.scraper --force    # re-scrape everything
```

Output: `data/articles/<topic>/<article-slug>.md`, one file per article, each with YAML frontmatter (`title`, `url`, `topic`, `topic_path`, `last_updated`, ...). Tables with merged cells are expanded into full markdown grids.

`data/manifest.json` tracks scraped articles; `data/failed.json` lists failures. A few sitemap entries redirect to a login page (non-public); these are skipped as `LoginRequired`.

## 2. Build the index

```powershell
python -m index.build_index                          # ~2 min on GPU
python -m index.search "power bank" --mode dense     # dense | bm25 | hybrid
```

## 3. Chat in the terminal

```powershell
python -m agent.chat --debug
```

## 4. Run the app

Terminal 1, the API (wait for `Application startup complete`):

```powershell
conda activate askmh
python -m uvicorn api.main:app --port 8001           # Swagger at http://127.0.0.1:8001/docs
```

Terminal 2, the React frontend (see [web/README.md](web/README.md)):

```powershell
cd web
npm install
npm run dev                                          # http://localhost:5173
```

API routes live under `/api` (`/api/health`, `/api/greeting`, `/api/chat`, `/api/feedback`, `/api/config`). `/go` stays at the root because source links open in a new tab without an auth header. If `web/dist` exists, the API also serves the built SPA at `/`, which is how the container runs everything on one port:

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

## 5. Tests

```powershell
python -m eval.run_suite                             # all six layers, ~70-100 s
python -m pytest tests/test_3_router.py -v           # one layer
```

Findings from the suite are written up in [eval/FINDINGS.md](eval/FINDINGS.md).
