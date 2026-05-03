"""Shared API access controls."""

from dataclasses import dataclass

import hashlib
import re

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.config import settings
from app.services import metadata_store

DEMO_SESSION_TTL_SECONDS = 24 * 3600
DEMO_TOKEN_PATTERN = re.compile(r"[A-Fa-f0-9]{64}")


@dataclass(frozen=True)
class AuthContext:
    id: str
    email: str
    display_name: str
    role: str
    is_system_admin: bool = False
    is_demo: bool = False
    demo_token: str = ""

    def as_user(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "is_system_admin": self.is_system_admin,
            "is_demo": self.is_demo,
            "demo_token": self.demo_token,
        }


SYSTEM_ADMIN = AuthContext(
    id="system",
    email="api-key",
    display_name="API key",
    role="admin",
    is_system_admin=True,
)


demo_router = APIRouter(prefix="/demo", tags=["Demo"])


def _normalize_demo_session_token(raw: str | None) -> str:
    if not isinstance(raw, str):
        return ""
    value = raw.strip()
    if not DEMO_TOKEN_PATTERN.fullmatch(value):
        return ""
    return value


def _demo_user(token: str) -> AuthContext:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    session_id = digest[:32]
    return AuthContext(
        id=f"demo:{session_id}",
        email=f"demo-{session_id[:8]}@public-demo.local",
        display_name="Public demo visitor",
        role="demo",
        is_demo=True,
        demo_token=token,
    )


@demo_router.post("/session", status_code=status.HTTP_201_CREATED)
async def create_demo_session() -> dict:
    if not settings.public_demo_mode:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public demo mode is disabled.",
        )
    if not hasattr(metadata_store, "create_demo_session"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo sessions are unavailable.",
        )
    return metadata_store.create_demo_session(ttl_seconds=DEMO_SESSION_TTL_SECONDS)


def _extract_bearer_token(authorization: str | None) -> str:
    if not isinstance(authorization, str):
        return ""
    bearer_prefix = "Bearer "
    if not authorization.startswith(bearer_prefix):
        return ""
    return authorization[len(bearer_prefix):].strip()


def _api_key_is_valid(x_api_key: str | None, authorization: str | None) -> bool:
    expected = settings.app_api_key.strip()
    if not expected:
        return False
    provided_api_key = x_api_key if isinstance(x_api_key, str) else None
    bearer_token = _extract_bearer_token(authorization)
    return provided_api_key == expected or bearer_token == expected


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Require an API key only when APP_API_KEY is configured."""
    expected = settings.app_api_key.strip()
    if not expected:
        return

    if _api_key_is_valid(x_api_key, authorization):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
    )


async def require_user(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
    x_demo_session_id: str | None = Header(default=None, alias="X-Demo-Session-Id"),
) -> AuthContext:
    """Require either a valid API key or a logged-in local user."""
    if not settings.enable_user_auth:
        if settings.app_api_key.strip():
            await require_api_key(x_api_key=x_api_key, authorization=authorization)
        return SYSTEM_ADMIN

    if _api_key_is_valid(x_api_key, authorization):
        return SYSTEM_ADMIN

    if settings.public_demo_mode:
        get_session = getattr(metadata_store, "get_demo_session", None)
        if get_session is None and isinstance(x_demo_session_id, str):
            legacy_test_token = x_demo_session_id.strip()
            if re.fullmatch(r"[A-Za-z0-9._:-]{16,128}", legacy_test_token):
                return _demo_user(legacy_test_token)

        demo_token = _normalize_demo_session_token(x_demo_session_id)
        if demo_token:
            if get_session(demo_token, ttl_seconds=DEMO_SESSION_TTL_SECONDS):
                return _demo_user(demo_token)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing, expired, or invalid demo session token.",
            )

    token = _extract_bearer_token(authorization)
    if token:
        user = metadata_store.get_user_for_token(token)
        if user:
            return AuthContext(
                id=str(user["id"]),
                email=str(user["email"]),
                display_name=str(user.get("display_name", "")),
                role=str(user.get("role", "user")),
            )

    detail = (
        "Create the first admin account at /auth/bootstrap."
        if not metadata_store.users_exist()
        else "Missing or invalid login token."
    )
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


async def require_admin(current_user: AuthContext = Depends(require_user)) -> AuthContext:
    if current_user.role == "admin" or current_user.is_system_admin:
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin role required.",
    )
