# =============================================================================
# auth.py — Identity resolution for the Clinical Decision Support API
# =============================================================================
"""
Two-mode actor resolution to support the Phase 5 transition:

  1. **JWT mode (preferred)** — `Authorization: Bearer <token>` header carries
     a Supabase-issued JWT. Signature is verified locally with HS256 and the
     `SUPABASE_JWT_SECRET` env var. The `sub` claim is the auth user id, which
     matches `users.id` in our application table; the role is looked up from
     that row (cached for `_ROLE_CACHE_TTL_SEC` seconds).

  2. **Header shim (fallback)** — `X-User-Id` + `X-User-Role` headers, set by
     the frontend's pre-auth dev role-switcher. Used when no Bearer token is
     present, or when `SUPABASE_JWT_SECRET` is not configured. This lets us
     ship 5b before 5c (frontend) is wired up.

Once the frontend always sends Bearer tokens, the header shim can be removed.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import jwt
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")
_JWT_AUDIENCE: str = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")
_JWT_ALGOS = ["HS256"]

if not _JWT_SECRET:
    logger.warning(
        "[auth] SUPABASE_JWT_SECRET not set — JWT verification disabled, "
        "falling back to X-User-Id / X-User-Role header shim. "
        "Set SUPABASE_JWT_SECRET in backend/.env to enable real auth."
    )

# Per-request role cache to avoid hitting the users table on every audited
# call. TTL is short (60s) so role/status changes propagate quickly.
_ROLE_CACHE: dict[str, tuple[str, float]] = {}  # user_id -> (role, expires_at)
_ROLE_CACHE_TTL_SEC = 60.0


def _verify_jwt(token: str) -> Optional[dict]:
    """Verify a Supabase HS256 JWT. Returns the claims dict, or None on failure."""
    if not _JWT_SECRET:
        return None
    try:
        return jwt.decode(
            token,
            _JWT_SECRET,
            algorithms=_JWT_ALGOS,
            audience=_JWT_AUDIENCE,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        logger.info("[auth] Rejected expired JWT")
        return None
    except jwt.InvalidTokenError as exc:
        logger.info("[auth] Rejected invalid JWT: %s", exc)
        return None


def _bearer_token(request: Request) -> Optional[str]:
    """Extract the Bearer token from Authorization, or None."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _lookup_role(user_id: str) -> Optional[str]:
    """Resolve role for a verified user_id from the users table, with TTL cache."""
    now = time.monotonic()
    cached = _ROLE_CACHE.get(user_id)
    if cached and cached[1] > now:
        return cached[0]

    # Lazy import — avoids pulling database.py at module-load time and keeps
    # this module testable in isolation.
    from database import get_db
    db = get_db()
    if db == "LOCAL_MOCK":
        # Mock mode has no users table; surface an empty role so callers can
        # decide policy. Header shim still works in this configuration.
        return None
    try:
        resp = (
            db.table("users")
            .select("role,status")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("[auth] users lookup failed for %s: %s", user_id, exc)
        return None

    if not resp.data:
        return None
    row = resp.data[0]
    if row.get("status") and row["status"] != "active":
        # Suspended/inactive — surface no role so _require_role denies access.
        logger.info("[auth] User %s has status=%s — denying", user_id, row["status"])
        return None
    role = row.get("role")
    if role:
        _ROLE_CACHE[user_id] = (role, now + _ROLE_CACHE_TTL_SEC)
    return role


def get_actor(request: Request) -> dict:
    """Return {'user_id', 'user_role'} for the calling request.

    Preference order:
      1. JWT in Authorization header (verified) → role looked up from users table
      2. X-User-Id + X-User-Role headers (dev shim) — trusted only if no JWT

    Both keys may be None if the caller is unauthenticated.
    """
    token = _bearer_token(request)
    if token:
        claims = _verify_jwt(token)
        if claims:
            user_id = claims.get("sub")
            # Prefer role from app_metadata if Supabase has been configured to
            # mint it into the JWT; otherwise fall back to the users table.
            role = (
                (claims.get("app_metadata") or {}).get("role")
                or (claims.get("user_metadata") or {}).get("role")
            )
            if not role and user_id:
                role = _lookup_role(user_id)
            return {"user_id": user_id, "user_role": role}

    # Fallback: dev header shim
    return {
        "user_id":   request.headers.get("X-User-Id") or None,
        "user_role": request.headers.get("X-User-Role") or None,
    }


def require_role(request: Request, *allowed: str) -> dict:
    """Server-side role gate. Returns the resolved actor on success; raises
    403 otherwise. Use as the first line of any role-restricted endpoint:

        actor = require_role(request, "system_admin")

    Combines identity resolution with policy check so callers can't forget
    to verify after extracting headers.
    """
    actor = get_actor(request)
    if actor.get("user_role") not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden — requires one of: {', '.join(allowed)}",
        )
    return actor
