from __future__ import annotations

import hmac
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping


@dataclass(frozen=True)
class AssistantUserContext:
    user_id: str
    username: str
    email: str


def canonical_user_context(user_id: str, username: str, email: str, timestamp: str) -> str:
    return "\n".join([user_id, username, email, timestamp])


def build_signature(secret: str, user_id: str, username: str, email: str, timestamp: str) -> str:
    payload = canonical_user_context(user_id, username, email, timestamp)
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), sha256).hexdigest()


def verify_user_context_headers(
    headers: Mapping[str, str],
    *,
    secret: str | None,
    ttl_seconds: int,
) -> AssistantUserContext:
    user_id = (headers.get("X-Assistant-User-Id") or headers.get("x-assistant-user-id") or "").strip()
    username = (headers.get("X-Assistant-Username") or headers.get("x-assistant-username") or "").strip()
    email = (headers.get("X-Assistant-User-Email") or headers.get("x-assistant-user-email") or "").strip()
    timestamp = (headers.get("X-Assistant-User-Timestamp") or headers.get("x-assistant-user-timestamp") or "").strip()
    signature = (headers.get("X-Assistant-User-Signature") or headers.get("x-assistant-user-signature") or "").strip()

    if not secret or not str(secret).strip():
        raise ValueError("Assistant user context signing secret is not configured.")

    if not user_id or not username or not email:
        raise ValueError("Missing assistant user context headers.")

    if not timestamp or not signature:
        raise ValueError("Missing signed assistant user context headers.")

    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise ValueError("Invalid assistant user context timestamp.") from exc

    now = int(time.time())
    if abs(now - ts) > max(1, int(ttl_seconds)):
        raise ValueError("Assistant user context timestamp is expired.")

    expected_signature = build_signature(secret, user_id, username, email, timestamp)
    if not hmac.compare_digest(expected_signature, signature):
        raise ValueError("Assistant user context signature mismatch.")

    return AssistantUserContext(user_id=user_id, username=username, email=email)
