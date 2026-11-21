"""Resolve the effective user identity for chat requests."""

from fastapi import Request

from app.config.settings import Settings
from app.core.exceptions import InvalidRequestError, UnauthorizedError


def resolve_user_id(request: Request, settings: Settings, body_user_id: str) -> str:
    """Body mode trusts user_id; trusted_header mode trusts only the gateway header."""
    if settings.auth_mode != "trusted_header":
        return body_user_id
    header_user = request.headers.get("x-user-id", "").strip()
    if not header_user:
        raise UnauthorizedError("missing X-User-Id")
    if len(header_user) > 64:
        raise InvalidRequestError("user id too long")
    return header_user
