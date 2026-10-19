import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import prompts
from agent.graph import build_graph
from agent.retriever import retrieve
from api.auth import User, check_auth_config, current_user, is_admin, require_admin, issue_session_token, verify_google_id_token
from api.limits import limiter
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    EndSessionRequest,
    FeedbackRequest,
    GoogleLoginRequest,
    GreetingResponse,
    HealthResponse,
    LoginResponse,
    Source,
)
from app import metrics
from app.metrics_sync import MetricsSync
from config import (
    AUTH_ENABLED,
    CHAT_MODEL,
    CHAT_PROVIDER,
    CORS_ORIGINS,
    GOOGLE_OAUTH_CLIENT_ID,
)

SOURCE_PREFIX = "https://www.askmh.malaysiaairlines.com/"
SCOPE = (
    "I answer only from the official AskMH FAQ and link the article I used. "
    "English only. Live chat runs 7:00 am-11:00 pm MYT."
)
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_auth_config()
    # Conversations live in RAM only and are deleted when the browser ends the session.
    app.state.checkpointer = MemorySaver()
    app.state.graph = build_graph(app.state.checkpointer)
    retrieve("warm up")
    sync = MetricsSync()
    sync.start()
    yield
    sync.stop()


app = FastAPI(title="AskMH API", version="1.1", lifespan=lifespan)
# Only needed when the SPA runs on its own origin (the Vite dev server). In the container
# the API serves the built SPA itself, so requests are same-origin.
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

api = APIRouter(prefix="/api")


@api.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", llm=f"{CHAT_PROVIDER}:{CHAT_MODEL}")


@api.get("/config", response_model=ConfigResponse)
def read_config():
    return ConfigResponse(auth_enabled=AUTH_ENABLED, google_client_id=GOOGLE_OAUTH_CLIENT_ID)


@api.post("/auth/google", response_model=LoginResponse)
def login_with_google(body: GoogleLoginRequest):
    user = verify_google_id_token(body.credential)
    token, expires_in = issue_session_token(user)
    logging.info("sign-in: %s", user.email or user.id)
    return LoginResponse(token=token, expires_in=expires_in, email=user.email, name=user.name)


@api.get("/greeting", response_model=GreetingResponse)
def greeting():
    return GreetingResponse(greeting=prompts.GREETING, scope=SCOPE)


def thread_id(user: User, session_id: str) -> str:
    """Conversations are keyed by owner too, so a session id sent by another account opens a
    fresh, empty conversation instead of someone else's history."""
    return f"{user.id}:{session_id}"


@api.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request, user: User = Depends(current_user)):
    session_id = body.session_id or uuid.uuid4().hex
    graph = request.app.state.graph
    # session_id in metadata lets LangSmith group a conversation's turns into one thread.
    config = {"configurable": {"thread_id": thread_id(user, session_id)}, "metadata": {"session_id": session_id}}

    unlimited = is_admin(user)
    with limiter.reserve(user.id, unlimited):
        before = len(graph.get_state(config).values.get("messages", []))
        t0 = time.time()
        try:
            state = graph.invoke({"messages": [HumanMessage(body.message)]}, config)
        except Exception as e:
            logging.exception("graph failed")
            metrics.log_message(session_id, body.message, "", "", "error", [], time.time() - t0)
            raise HTTPException(
                status_code=503, detail="Assistant temporarily unavailable, try again shortly"
            ) from e
        latency = time.time() - t0

    replies = [m.content for m in state["messages"][before + 1:] if isinstance(m, AIMessage)]
    outcome = metrics.outcome_of(state, len(replies))
    urls = state.get("sources", []) if outcome == "answered" else []
    message_id = metrics.log_message(
        session_id, state.get("question", body.message), state["route"], state.get("decision", ""),
        outcome, urls, latency,
    )

    sources: list[Source] = []
    if outcome == "answered":
        replies[-1] = replies[-1].split("\n\nSources:")[0]
        titles = {d.metadata["url"]: d.metadata["title"] for d in state["docs"]}
        for url in urls:
            click_url = request.url_for("go").include_query_params(u=url, s=session_id, m=message_id)
            sources.append(Source(title=titles.get(url, url), url=url, click_url=str(click_url)))

    return ChatResponse(
        session_id=session_id, message_id=message_id, replies=replies, sources=sources,
        route=state["route"], decision=state.get("decision", ""), outcome=outcome,
        latency_s=round(latency, 2), messages_left=limiter.remaining(user.id, unlimited),
    )


@api.post("/feedback", status_code=204)
def feedback(body: FeedbackRequest, user: User = Depends(current_user)):
    metrics.log_feedback(body.session_id, body.message_id, body.value)


@api.post("/session/end", status_code=204)
def end_session(body: EndSessionRequest, request: Request, user: User = Depends(current_user)):
    request.app.state.checkpointer.delete_thread(thread_id(user, body.session_id))


@api.get("/admin/stats")
def admin_stats(days: int = Query(14, ge=1, le=365), user: User = Depends(require_admin)):
    return metrics.stats(days)


app.include_router(api)


# Stays unauthenticated and at the root: source links open in a new tab, which carries no
# Authorization header. It only logs a click and redirects to an allowlisted domain.
@app.get("/go", name="go")
def go(u: str, s: str = "", m: int = 0):
    if not u.startswith(SOURCE_PREFIX):
        raise HTTPException(status_code=400, detail="Unsupported link")
    metrics.log_click(s, m, u)
    return RedirectResponse(u)


# Mounted last so it cannot shadow the API routes. Absent during API-only local dev,
# where Vite serves the frontend instead.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="spa")
