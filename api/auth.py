"""Google sign-in for the hosted demo.

The browser gets an ID token from Google, we verify it, then issue our own short-lived
session JWT. The Gemini key never leaves the server; sign-in exists to protect the shared
free-tier quota and to record who used the demo.

With AUTH_ENABLED off (the default) every caller is the same anonymous user, so local dev
and the test suite need no credentials.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from config import ADMIN_EMAILS, AUTH_ENABLED, GOOGLE_OAUTH_CLIENT_ID, SESSION_SECRET, SESSION_TTL_HOURS

ALGORITHM = "HS256"
ANONYMOUS = "anonymous"
DEV_SECRET = "dev-only-secret"
MIN_SECRET_BYTES = 32


def check_auth_config() -> None:
    """Fail at boot rather than serve a gate anyone can forge a token for.

    HS256 tokens are only as trustworthy as the secret, so shipping the default would
    let anyone mint a valid session and bypass sign-in entirely.
    """
    if not AUTH_ENABLED:
        return
    if not GOOGLE_OAUTH_CLIENT_ID:
        raise RuntimeError("AUTH_ENABLED is on but GOOGLE_OAUTH_CLIENT_ID is not set")
    if SESSION_SECRET == DEV_SECRET or len(SESSION_SECRET) < MIN_SECRET_BYTES:
        raise RuntimeError(
            f"AUTH_ENABLED is on but SESSION_SECRET is the default or shorter than "
            f"{MIN_SECRET_BYTES} characters. Generate one with: "
            f"python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

# auto_error=False so we can return our own message when auth is enabled but no token was sent.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class User:
    id: str
    email: str = ""
    name: str = ""


def verify_google_id_token(credential: str) -> User:
    """Validate the ID token Google handed the browser."""
    if not GOOGLE_OAUTH_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Server is missing GOOGLE_OAUTH_CLIENT_ID")
    try:
        claims = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_OAUTH_CLIENT_ID
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail="Google sign-in could not be verified") from e
    # An unverified email must not match ADMIN_EMAILS.
    email = claims.get("email", "") if claims.get("email_verified") else ""
    return User(id=claims["sub"], email=email, name=claims.get("name", ""))


def issue_session_token(user: User) -> tuple[str, int]:
    """Our own token, so the Google credential is not replayed on every request."""
    expires_in = SESSION_TTL_HOURS * 3600
    payload = {
        "sub": user.id,
        "email": user.email,
        "name": user.name,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
    }
    return jwt.encode(payload, SESSION_SECRET, algorithm=ALGORITHM), expires_in


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if not AUTH_ENABLED:
        return User(id=ANONYMOUS)
    if creds is None:
        raise HTTPException(status_code=401, detail="Sign in to use the demo")
    try:
        claims = jwt.decode(creds.credentials, SESSION_SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail="Session expired, sign in again") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="Invalid session") from e
    return User(id=claims["sub"], email=claims.get("email", ""), name=claims.get("name", ""))


def is_admin(user: User) -> bool:
    return bool(user.email) and user.email.lower() in ADMIN_EMAILS


def require_admin(user: User = Depends(current_user)) -> User:
    """With sign-in off (local dev) everyone is admin; when hosted only ADMIN_EMAILS are."""
    if AUTH_ENABLED and not is_admin(user):
        raise HTTPException(status_code=403, detail="Admins only")
    return user
