# AskMH web

React + TypeScript + Vite + Tailwind chat frontend for the AskMH FastAPI backend (`api/`).

## Run

```powershell
# 1. API (separate terminal, repo root)
conda activate askmh
python -m uvicorn api.main:app --port 8001

# 2. Frontend (this folder)
npm install
npm run dev          # http://localhost:5173
```

The API URL defaults to `http://127.0.0.1:8001`; override with `VITE_API_URL` in `web/.env` (see `.env.example`). The dev server must stay on port 5173, the origin the API's `CORS_ORIGINS` allows.

## Scripts

| Command | Does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Type-check (`tsc -b`) and build to `dist/` |
| `npm run lint` | oxlint |
| `npm run preview` | Serve the built `dist/` |

## Layout

```
src/api.ts                       typed client for /greeting, /chat, /feedback
src/useChat.ts                   chat state, localStorage persistence, send / rate / new chat
src/App.tsx                      page layout, debug toggle, auto-scroll
src/components/Message.tsx       bubble: markdown, sources, feedback, debug line
src/components/Composer.tsx      input box
src/components/TypingIndicator.tsx
```
