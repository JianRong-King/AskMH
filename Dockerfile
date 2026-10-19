# Single container for Cloud Run (or an HF Space): FastAPI serves the API and the built SPA
# on one port, so there is no CORS and no second process.

# ---- stage 1: build the React bundle ----
FROM node:22-slim AS web
WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- stage 2: runtime ----
FROM python:3.11-slim

# Runs as uid 1000 (what HF Spaces expect). HF_HOME must point somewhere that user can write.
RUN useradd -m -u 1000 app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# CPU torch only. The host has no GPU and the CUDA wheel is several GB larger.
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY --chown=app:app . .
COPY --from=web --chown=app:app /build/dist ./web/dist

USER app

# Bake the embedding model into the image, so waking from sleep costs a model load
# rather than a 440 MB download.
RUN python -c "from index.embeddings import get_embeddings; get_embeddings()"

ENV CHAT_PROVIDER=gemini \
    AUTH_ENABLED=true \
    EMBED_DEVICE=cpu

EXPOSE 7860

# One worker on purpose: the quota counters in api/limits.py are in-process.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
