# =============================================================================
# auth.py — Identity resolution for the Clinical Decision Support API
# =============================================================================
"""
Two-mode actor resolution:

  1. **JWT mode (preferred)** — `Authorization: Bearer <token>` carries a
     Supabase-issued JWT. Verification is delegated to Supabase via
     `auth.get_user(token)` so it works with both legacy HS256 and the
     current ECC P-256 signing key — no SUPABASE_JWT_SECRET needed.
     The returned email bridges to our `users` table (since auth.uid()
     differs from our seeded users.id UUIDs). Role is cached for 60s.

  2. **Header shim (fallback)** — `X-User-Id` + `X-User-Role` headers, set
     by the frontend dev role-switcher when no Bearer token is present.
     Kept for local dev so the site works without a live Supabase session.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# email -> (app_user_id, role, expires_at)
_ROLE_CACHE: dict[str, tuple[str, str, float]] = {}
_ROLE_CACHE_TTL_SEC = 60.0


def _bearer_token(request: Request) -> Optional[str]:
    """Extract the Bearer token from Authorization, or None."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _verify_token_via_supabase(token: str) -> Optional[str]:
    """Call Supabase to verify the token and return the user's email, or None.

    Uses the service-role Supabase client so it can call auth.get_user()
    regardless of RLS policies. Works with HS256 and ECC P-256 signing keys.
    """
    try:
        from database import get_db
        db = get_db()
        if db == "LOCAL_MOCK":
            return None
        resp = db.auth.get_user(token)
        user = resp.user if hasattr(resp, "user") else None
        if user and user.email:
            return user.email
        return None
    except Exception as exc:
        logger.info("[auth] Token verification failed: %s", exc)
        return None


def _lookup_by_email(email: str) -> Optional[tuple[str, str]]:
    """Return (app_user_id, role) from our users table by email, with TTL cache."""
    now = time.monotonic()
    cached = _ROLE_CACHE.get(email)
    if cached and cached[2] > now:
        return (cached[0], cached[1])

    from database import get_db
    db = get_db()
    if db == "LOCAL_MOCK":
        return None
    try:
        resp = (
            db.table("users")
            .select("id,role,status")
            .eq("email", email)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.warning("[auth] users email lookup failed for %s: %s", email, exc)
        return None

    if not resp.data:
        logger.info("[auth] No users row for email=%s", email)
        return None
    row = resp.data[0]
    if row.get("status") and row["status"] != "active":
        logger.info("[auth] User %s status=%s — denying", email, row["status"])
        return None
    role = row.get("role")
    app_user_id = row.get("id")
    if role and app_user_id:
        _ROLE_CACHE[email] = (app_user_id, role, now + _ROLE_CACHE_TTL_SEC)
        return (app_user_id, role)
    return None


def get_actor(request: Request) -> dict:
    """Return {'user_id', 'user_role'} for the calling request.

    Preference order:
      1. Bearer JWT → verified via Supabase → email → users table lookup
      2. X-User-Id + X-User-Role dev shim — ONLY when no Bearer token is
         present at all. If a Bearer token is supplied but invalid or its
         email isn't in our users table, we return an empty actor and do
         NOT fall through to the shim. Otherwise an attacker could attach
         any expired/fake token alongside spoofed X-User-Role headers and
         escalate via the shim.
    """
    token = _bearer_token(request)
    if token:
        email = _verify_token_via_supabase(token)
        if not email:
            # Token present but invalid/expired/unverifiable — reject.
            # Do not honour the dev shim when a token was attempted.
            return {"user_id": None, "user_role": None}
        result = _lookup_by_email(email)
        if not result:
            # Authenticated identity, but no matching active users row.
            return {"user_id": None, "user_role": None}
        app_user_id, role = result
        return {"user_id": app_user_id, "user_role": role}

    # No Bearer token at all → dev header shim (local dev only).
    return {
        "user_id":   request.headers.get("X-User-Id") or None,
        "user_role": request.headers.get("X-User-Role") or None,
    }


def require_role(request: Request, *allowed: str) -> dict:
    """Role gate — raises 403 if the verified role isn't in `allowed`.

    Usage:
        actor = require_role(request, "system_admin")
    """
    actor = get_actor(request)
    if actor.get("user_role") not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden — requires one of: {', '.join(allowed)}",
        )
    return actor
