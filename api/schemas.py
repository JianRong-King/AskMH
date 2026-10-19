from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Omit to start a new conversation")


class Source(BaseModel):
    title: str
    url: str
    click_url: str


class ChatResponse(BaseModel):
    session_id: str
    message_id: int
    replies: list[str]
    sources: list[Source]
    route: str
    decision: str
    outcome: Literal["answered", "clarify", "declined", "handoff", "greeting"]
    latency_s: float
    messages_left: int | None = Field(description="Remaining messages for this user today; null when unlimited")


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: int
    value: Literal[1, -1]


class EndSessionRequest(BaseModel):
    session_id: str


class GreetingResponse(BaseModel):
    greeting: str
    scope: str


class HealthResponse(BaseModel):
    status: str
    llm: str


class ConfigResponse(BaseModel):
    """Lets the frontend discover whether to show sign-in, so the client id is not baked
    into the build and one image can serve different deployments."""

    auth_enabled: bool
    google_client_id: str


class GoogleLoginRequest(BaseModel):
    credential: str = Field(min_length=1, description="ID token from Google Identity Services")


class LoginResponse(BaseModel):
    token: str
    expires_in: int
    email: str
    name: str
