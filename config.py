import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
ARTICLES_DIR = DATA_DIR / "articles"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "askmh"
METRICS_DB = DATA_DIR / "metrics.db"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_QUERY_INSTRUCTION = os.getenv(
    "EMBED_QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
)
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "")

CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "ollama")  # "ollama" or "gemini"
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b" if CHAT_PROVIDER == "ollama" else "gemini-2.5-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_RPM = float(os.getenv("LLM_RPM", "4"))
# Gemini 2.5 thinking tokens: 0 = off (matches local Qwen with reasoning=False), -1 = dynamic.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
LLM_SEED = int(os.getenv("LLM_SEED", "42"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Google sign-in. Off by default so local dev and the test suite need no credentials.
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-only-secret")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "12"))

# Quota guards. The hosted demo shares one free-tier key, so these keep it usable
# rather than prevent a bill; the free tier cannot be charged.
DAILY_MESSAGES_PER_USER = int(os.getenv("DAILY_MESSAGES_PER_USER", "10"))
DAILY_MESSAGES_GLOBAL = int(os.getenv("DAILY_MESSAGES_GLOBAL", "300"))

# Monitoring. ADMIN_EMAILS may open /#admin when sign-in is on. METRICS_REPO is a private
# HF Dataset that metrics.db is mirrored to, since a free Space's disk resets on restart.
ADMIN_EMAILS = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
METRICS_REPO = os.getenv("METRICS_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
METRICS_SYNC_MINUTES = float(os.getenv("METRICS_SYNC_MINUTES", "10"))

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1500"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "5"))
