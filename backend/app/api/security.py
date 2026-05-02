"""Shared API access controls."""

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from app.config import settings
from app.services import metadata_store


@dataclass(frozen=True)
class AuthContext:
    id: str
    email: str
    display_name: str
    role: str
    is_system_admin: bool = False

    def as_user(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "is_system_admin": self.is_system_admin,
        }


SYSTEM_ADMIN = AuthContext(
    id="system",
    email="api-key",
    display_name="API key",
    role="admin",
    is_system_admin=True,
)


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
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> AuthContext:
    """Require either a valid API key or a logged-in local user."""
    if not settings.enable_user_auth:
        if settings.app_api_key.strip():
            await require_api_key(x_api_key=x_api_key, authorization=authorization)
        return SYSTEM_ADMIN

    if _api_key_is_valid(x_api_key, authorization):
        return SYSTEM_ADMIN

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
