"""Shared API access controls."""

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Require an API key only when APP_API_KEY is configured."""
    expected = settings.app_api_key.strip()
    if not expected:
        return

    provided_api_key = x_api_key if isinstance(x_api_key, str) else None
    provided_authorization = authorization if isinstance(authorization, str) else None

    bearer_prefix = "Bearer "
    bearer_token = ""
    if provided_authorization and provided_authorization.startswith(bearer_prefix):
        bearer_token = provided_authorization[len(bearer_prefix):].strip()

    if provided_api_key == expected or bearer_token == expected:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
    )
