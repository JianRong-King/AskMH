import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

from config import METRICS_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT NOT NULL,
    question TEXT,
    route TEXT,
    decision TEXT,
    outcome TEXT,
    sources TEXT,
    latency_s REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_id INTEGER,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS clicks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT,
    message_id INTEGER,
    url TEXT NOT NULL
);
"""


def outcome_of(state: dict, n_replies: int) -> str:
    if state["route"] == "greeting":
        return "greeting"
    if state["route"] == "human":
        return "handoff"
    if state["decision"] == "clarify":
        return "clarify"
    # The grader can say "answer" and the generator still refuse, clearing sources.
    # That is a decline, not an answered turn, or the metrics overcount resolutions.
    if state["decision"] == "answer" and state.get("sources"):
        return "answered"
    return "handoff" if n_replies > 1 else "declined"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(sql: str, params: tuple = ()) -> int:
    METRICS_DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(METRICS_DB)) as db:
        db.executescript(SCHEMA)
        cur = db.execute(sql, params)
        db.commit()
        return cur.lastrowid


def log_message(session_id: str, question: str, route: str, decision: str, outcome: str,
                sources: list[str], latency_s: float) -> int:
    return _run(
        "INSERT INTO messages (ts, session_id, question, route, decision, outcome, sources, latency_s) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_now(), session_id, question, route, decision, outcome, json.dumps(sources), round(latency_s, 2)),
    )


def log_feedback(session_id: str, message_id: int, value: int) -> None:
    _run("INSERT INTO feedback (ts, session_id, message_id, value) VALUES (?, ?, ?, ?)",
         (_now(), session_id, message_id, value))


def log_click(session_id: str, message_id: int, url: str) -> None:
    _run("INSERT INTO clicks (ts, session_id, message_id, url) VALUES (?, ?, ?, ?)",
         (_now(), session_id, message_id, url))


def stats(days: int = 14) -> dict:
    """Dashboard numbers for the last `days` days. Timestamps are UTC ISO strings."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    METRICS_DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(METRICS_DB)) as db:
        db.executescript(SCHEMA)
        db.row_factory = sqlite3.Row

        def one(sql: str) -> sqlite3.Row:
            return db.execute(sql, (since,)).fetchone()

        def rows(sql: str) -> list[dict]:
            return [dict(r) for r in db.execute(sql, (since,))]

        totals = one(
            "SELECT COUNT(*) AS messages, COUNT(DISTINCT session_id) AS sessions, "
            "ROUND(AVG(CASE WHEN outcome != 'error' THEN latency_s END), 2) AS avg_latency_s "
            "FROM messages WHERE ts >= ?"
        )
        outcomes = {r["outcome"]: r["n"] for r in rows(
            "SELECT outcome, COUNT(*) AS n FROM messages WHERE ts >= ? GROUP BY outcome"
        )}
        votes = one("SELECT COUNT(*) AS n, SUM(value = 1) AS up FROM feedback WHERE ts >= ?")
        clicked = one("SELECT COUNT(DISTINCT message_id) AS n FROM clicks WHERE ts >= ?")["n"]

        # Greetings and server errors are not attempts to answer, so they do not count.
        attempts = sum(n for o, n in outcomes.items() if o not in ("greeting", "error"))
        answered = outcomes.get("answered", 0)
        ratio = lambda a, b: round(a / b, 3) if b else None  # noqa: E731

        return {
            "days": days,
            "messages": totals["messages"],
            "sessions": totals["sessions"],
            "avg_latency_s": totals["avg_latency_s"],
            "outcomes": outcomes,
            "resolution_rate": ratio(answered, attempts),
            "csat": ratio(votes["up"] or 0, votes["n"]),
            "feedback_count": votes["n"],
            "source_ctr": ratio(clicked, answered),
            "per_day": rows(
                "SELECT substr(ts, 1, 10) AS day, COUNT(*) AS messages FROM messages "
                "WHERE ts >= ? GROUP BY day ORDER BY day"
            ),
            "recent": rows(
                "SELECT ts, question, route, outcome, latency_s FROM messages "
                "WHERE ts >= ? AND outcome != 'greeting' ORDER BY id DESC LIMIT 50"
            ),
            "unanswered": rows(
                "SELECT ts, question, outcome FROM messages "
                "WHERE ts >= ? AND outcome IN ('declined', 'handoff') ORDER BY id DESC LIMIT 20"
            ),
        }
