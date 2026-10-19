"""Quota guards for the hosted demo.

Everyone shares one free-tier Gemini key at 5 requests/minute, and an FAQ turn costs three
calls (router, grader, generator). So the real scarce resource is the key, not money.

Counters are in memory on purpose: Cloud Run's disk is ephemeral, so persisting them would
buy nothing. This assumes a single instance with one uvicorn worker.

Unlimited users (ADMIN_EMAILS) skip the daily caps and do not use up the global one, but
still get only one message in flight.
"""

import threading
from contextlib import contextmanager
from datetime import date

from fastapi import HTTPException

from config import DAILY_MESSAGES_GLOBAL, DAILY_MESSAGES_PER_USER

BUSY = "You already have a message in flight. Wait for it to finish."
USER_CAP = (
    "You have reached today's message limit for this account. To keep chatting, use Sign out "
    "(top right) and sign in with another Google account, or try again tomorrow."
)
GLOBAL_CAP = "The demo has hit its shared daily limit. Try again tomorrow."


class Limiter:
    def __init__(self, per_user: int, globally: int):
        self.per_user = per_user
        self.globally = globally
        self._lock = threading.Lock()
        self._day = date.today()
        self._per_user_today: dict[str, int] = {}
        self._total_today = 0
        self._in_flight: set[str] = set()

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._per_user_today.clear()
            self._total_today = 0

    @contextmanager
    def reserve(self, user_id: str, unlimited: bool = False):
        """Count one message against the caps, and hold a per-user in-flight slot."""
        with self._lock:
            self._roll_day()
            if user_id in self._in_flight:
                raise HTTPException(status_code=429, detail=BUSY)
            if not unlimited:
                if self._total_today >= self.globally:
                    raise HTTPException(status_code=429, detail=GLOBAL_CAP)
                if self._per_user_today.get(user_id, 0) >= self.per_user:
                    raise HTTPException(status_code=429, detail=USER_CAP)
                self._per_user_today[user_id] = self._per_user_today.get(user_id, 0) + 1
                self._total_today += 1
            self._in_flight.add(user_id)
        try:
            yield
        finally:
            with self._lock:
                self._in_flight.discard(user_id)

    def remaining(self, user_id: str, unlimited: bool = False) -> int | None:
        if unlimited:
            return None
        with self._lock:
            self._roll_day()
            return max(0, self.per_user - self._per_user_today.get(user_id, 0))


limiter = Limiter(DAILY_MESSAGES_PER_USER, DAILY_MESSAGES_GLOBAL)
