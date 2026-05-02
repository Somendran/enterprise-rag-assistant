"""Local user authentication endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.security import AuthContext, require_admin, require_user
from app.config import settings
from app.models.schemas import (
    AuthBootstrapRequest,
    AuthLoginRequest,
    AuthStatusResponse,
    AuthTokenResponse,
    CurrentUserResponse,
    UserCreateRequest,
    UserItem,
    UsersResponse,
)
from app.services import metadata_store

router = APIRouter(prefix="/auth", tags=["Auth"])


def _token_response(user: dict, token_payload: dict) -> AuthTokenResponse:
    return AuthTokenResponse(
        access_token=str(token_payload["access_token"]),
        token_type=str(token_payload.get("token_type", "bearer")),
        expires_at=int(token_payload["expires_at"]),
        user=UserItem(**user),
    )


def _record(actor: AuthContext | dict | None, action: str, detail: dict | None = None) -> None:
    if isinstance(actor, AuthContext):
        actor_id = actor.id
        actor_email = actor.email
    elif isinstance(actor, dict):
        actor_id = str(actor.get("id", ""))
        actor_email = str(actor.get("email", ""))
    else:
        actor_id = ""
        actor_email = ""
    metadata_store.record_audit_event(
        actor_user_id=actor_id,
        actor_email=actor_email,
        action=action,
        resource_type="auth",
        detail=detail or {},
    )


@router.get("/status", response_model=AuthStatusResponse, status_code=status.HTTP_200_OK)
async def auth_status() -> AuthStatusResponse:
    has_users = metadata_store.users_exist()
    return AuthStatusResponse(
        auth_enabled=bool(settings.enable_user_auth),
        has_users=has_users,
        bootstrap_required=bool(settings.enable_user_auth and not has_users),
    )


@router.post("/bootstrap", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
async def bootstrap_admin(request: AuthBootstrapRequest) -> AuthTokenResponse:
    if metadata_store.users_exist():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The first admin account already exists.",
        )
    user = metadata_store.create_user(
        user_id=uuid4().hex,
        email=request.email,
        password=request.password,
        display_name=request.display_name,
        role="admin",
    )
    token = metadata_store.create_auth_token(
        str(user["id"]),
        ttl_seconds=int(settings.auth_token_ttl_hours) * 3600,
    )
    _record(user, "auth.bootstrap", {"email": user.get("email")})
    return _token_response(user, token)


@router.post("/login", response_model=AuthTokenResponse, status_code=status.HTTP_200_OK)
async def login(request: AuthLoginRequest) -> AuthTokenResponse:
    user = metadata_store.authenticate_user(request.email, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    token = metadata_store.create_auth_token(
        str(user["id"]),
        ttl_seconds=int(settings.auth_token_ttl_hours) * 3600,
    )
    _record(user, "auth.login", {"email": user.get("email")})
    return _token_response(user, token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: AuthContext = Depends(require_user),
    authorization: str | None = Header(default=None),
) -> None:
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):].strip()
        if token:
            metadata_store.revoke_auth_token(token)
    _record(current_user, "auth.logout")


@router.get("/me", response_model=CurrentUserResponse, status_code=status.HTTP_200_OK)
async def me(current_user: AuthContext = Depends(require_user)) -> CurrentUserResponse:
    stored_user = metadata_store.get_user(current_user.id)
    if stored_user:
        return CurrentUserResponse(user=UserItem(**stored_user))
    return CurrentUserResponse(
        user=UserItem(
            id=current_user.id,
            email=current_user.email,
            display_name=current_user.display_name,
            role=current_user.role,
            disabled=0,
            created_at=0,
            updated_at=0,
        )
    )


@router.get("/users", response_model=UsersResponse, status_code=status.HTTP_200_OK)
async def list_users(_: AuthContext = Depends(require_admin)) -> UsersResponse:
    return UsersResponse(users=[UserItem(**item) for item in metadata_store.list_users()])


@router.post("/users", response_model=UserItem, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: UserCreateRequest,
    current_user: AuthContext = Depends(require_admin),
) -> UserItem:
    if metadata_store.get_user_by_email(request.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists.")
    user = metadata_store.create_user(
        user_id=uuid4().hex,
        email=request.email,
        password=request.password,
        display_name=request.display_name,
        role=request.role,
    )
    _record(current_user, "user.create", {"email": user.get("email"), "role": user.get("role")})
    return UserItem(**user)
