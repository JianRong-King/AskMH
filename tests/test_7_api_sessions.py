"""Layer 7: conversations belong to the account that started them.

Uses the real FastAPI app with a stub graph (no LLM) and a fresh in-memory checkpointer. The app's
lifespan is not run, so no model loads and metrics writes are stubbed out.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import State
from api import main
from api.auth import User, current_user


def _stub_graph(checkpointer):
    def reply(state: State) -> State:
        return {"route": "greeting", "messages": [AIMessage(f"turn {len(state['messages'])}")]}

    g = StateGraph(State)
    g.add_node("reply", reply)
    g.add_edge(START, "reply")
    g.add_edge("reply", END)
    return g.compile(checkpointer=checkpointer)


@pytest.fixture
def api(monkeypatch):
    checkpointer = MemorySaver()
    main.app.state.checkpointer = checkpointer
    main.app.state.graph = _stub_graph(checkpointer)
    monkeypatch.setattr(main.metrics, "log_message", lambda *a, **k: 1)
    as_user = {"id": "alice"}
    main.app.dependency_overrides[current_user] = lambda: User(id=as_user["id"])
    yield TestClient(main.app), as_user, checkpointer
    main.app.dependency_overrides.clear()


def _history(checkpointer, user_id: str, session_id: str) -> int:
    state = checkpointer.get({"configurable": {"thread_id": f"{user_id}:{session_id}", "checkpoint_ns": ""}})
    return len(state["channel_values"]["messages"]) if state else 0


def test_other_account_cannot_continue_a_conversation(api):
    client, as_user, checkpointer = api
    session = client.post("/api/chat", json={"message": "hi"}).json()["session_id"]
    client.post("/api/chat", json={"message": "again", "session_id": session})
    assert _history(checkpointer, "alice", session) == 4

    as_user["id"] = "mallory"
    reply = client.post("/api/chat", json={"message": "what did they ask?", "session_id": session}).json()
    assert reply["replies"] == ["turn 1"], "mallory saw alice's history"
    assert _history(checkpointer, "alice", session) == 4
    assert _history(checkpointer, "mallory", session) == 2


def test_other_account_cannot_end_a_conversation(api):
    client, as_user, checkpointer = api
    session = client.post("/api/chat", json={"message": "hi"}).json()["session_id"]

    as_user["id"] = "mallory"
    assert client.post("/api/session/end", json={"session_id": session}).status_code == 204
    assert _history(checkpointer, "alice", session) == 2

    as_user["id"] = "alice"
    client.post("/api/session/end", json={"session_id": session})
    assert _history(checkpointer, "alice", session) == 0
