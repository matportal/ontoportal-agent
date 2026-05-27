from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests

logger = logging.getLogger("ontoportal_agent.server")

CODEX_DEVICE_URL = "https://auth.openai.com/codex/device"
_DEFAULT_ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
)


class AntigravityConfigError(RuntimeError):
    """Raised when shared Gemini Antigravity OAuth app credentials are not configured."""


def _env_value(env_name: str) -> str:
    return str(os.getenv(env_name) or "").strip()


def antigravity_redirect_uri() -> str:
    return _env_value("ONTOAGENT_ANTIGRAVITY_REDIRECT_URI") or _DEFAULT_ANTIGRAVITY_REDIRECT_URI


def require_antigravity_oauth_config() -> tuple[str, str, str]:
    client_id = _env_value("ONTOAGENT_ANTIGRAVITY_CLIENT_ID")
    client_secret = _env_value("ONTOAGENT_ANTIGRAVITY_CLIENT_SECRET")
    redirect_uri = antigravity_redirect_uri()
    if not client_id or not client_secret:
        raise AntigravityConfigError(
            "Gemini Antigravity login is not configured. Set ONTOAGENT_ANTIGRAVITY_CLIENT_ID "
            "and ONTOAGENT_ANTIGRAVITY_CLIENT_SECRET from the Kubernetes secret before starting login."
        )
    return client_id, client_secret, redirect_uri


def _antigravity_config_source(env_name: str) -> str:
    if env_name == "ONTOAGENT_ANTIGRAVITY_REDIRECT_URI" and not os.getenv(env_name):
        return "default"
    return "env" if os.getenv(env_name) else "missing"


def antigravity_oauth_config_summary() -> dict[str, Any]:
    client_id = _env_value("ONTOAGENT_ANTIGRAVITY_CLIENT_ID")
    return {
        "client_id_suffix": client_id[-18:] if client_id else "",
        "client_id_hash": hashlib.sha256(client_id.encode()).hexdigest()[:12] if client_id else "",
        "client_id_source": _antigravity_config_source("ONTOAGENT_ANTIGRAVITY_CLIENT_ID"),
        "client_secret_configured": bool(_env_value("ONTOAGENT_ANTIGRAVITY_CLIENT_SECRET")),
        "client_secret_source": _antigravity_config_source("ONTOAGENT_ANTIGRAVITY_CLIENT_SECRET"),
        "redirect_uri": antigravity_redirect_uri(),
        "redirect_uri_source": _antigravity_config_source("ONTOAGENT_ANTIGRAVITY_REDIRECT_URI"),
        "scope_count": len(ANTIGRAVITY_SCOPES),
    }


@dataclass
class AuthSession:
    id: str
    user_id: str
    provider: str
    expires_at: datetime
    login_url: str = ""
    user_code: str = ""
    state: str = ""
    code_verifier: str = ""
    project_id: str = ""
    temp_dir: Path | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    output_lines: list[str] = field(default_factory=list, repr=False)
    error: str = ""


class AccountAuthManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, AuthSession] = {}

    def cleanup_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired: list[AuthSession] = []
        with self._lock:
            for session_id, auth_session in list(self._sessions.items()):
                if auth_session.expires_at <= now:
                    expired.append(auth_session)
                    self._sessions.pop(session_id, None)
        for auth_session in expired:
            self._cleanup_session(auth_session)

    def get(self, *, session_id: str, user_id: str, provider: str | None = None) -> AuthSession:
        self.cleanup_expired()
        with self._lock:
            auth_session = self._sessions.get(session_id)
        if auth_session is None or auth_session.user_id != user_id:
            raise KeyError("Auth session was not found.")
        if provider and auth_session.provider != provider:
            raise KeyError("Auth session provider does not match.")
        return auth_session

    def pop(self, *, session_id: str, user_id: str, provider: str | None = None) -> AuthSession:
        auth_session = self.get(session_id=session_id, user_id=user_id, provider=provider)
        with self._lock:
            self._sessions.pop(session_id, None)
        return auth_session

    def start_codex_device_auth(self, *, user_id: str, codex_path: str = "codex") -> AuthSession:
        self.cleanup_expired()
        session_id = uuid.uuid4().hex
        temp_dir = Path(tempfile.mkdtemp(prefix="matportal-codex-auth-"))
        temp_dir.chmod(0o700)
        env = dict(os.environ)
        env["CODEX_HOME"] = str(temp_dir)
        process = subprocess.Popen(
            [codex_path, "login", "--device-auth"],
            cwd=str(temp_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        auth_session = AuthSession(
            id=session_id,
            user_id=user_id,
            provider="codex",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            temp_dir=temp_dir,
            process=process,
            login_url=CODEX_DEVICE_URL,
        )
        threading.Thread(target=self._read_codex_output, args=(auth_session,), daemon=True).start()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self._refresh_codex_fields(auth_session)
            if auth_session.user_code:
                break
            if process.poll() is not None:
                self._refresh_codex_fields(auth_session)
                break
            time.sleep(0.1)
        with self._lock:
            self._sessions[session_id] = auth_session
        return auth_session

    def start_antigravity(self, *, user_id: str, project_id: str = "") -> AuthSession:
        self.cleanup_expired()
        client_id, _client_secret, redirect_uri = require_antigravity_oauth_config()
        code_verifier = _pkce_verifier()
        state = secrets.token_urlsafe(24)
        config_summary = antigravity_oauth_config_summary()
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(ANTIGRAVITY_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": _pkce_challenge(code_verifier),
            "code_challenge_method": "S256",
        }
        auth_session = AuthSession(
            id=uuid.uuid4().hex,
            user_id=user_id,
            provider="gemini_antigravity",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            login_url=f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}",
            state=state,
            code_verifier=code_verifier,
            project_id=str(project_id or ""),
        )
        with self._lock:
            self._sessions[auth_session.id] = auth_session
        logger.info(
            "Gemini Antigravity auth session started user=%s session=%s project_set=%s oauth=%s",
            user_id,
            auth_session.id,
            bool(auth_session.project_id),
            config_summary,
        )
        return auth_session

    def codex_auth_json_path(self, auth_session: AuthSession) -> Path | None:
        if auth_session.temp_dir is None:
            return None
        path = auth_session.temp_dir / "auth.json"
        if path.exists() and path.is_file():
            return path
        return None

    def finish(self, auth_session: AuthSession) -> None:
        self._cleanup_session(auth_session)

    def _read_codex_output(self, auth_session: AuthSession) -> None:
        process = auth_session.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                clean = str(line or "").strip()
                if clean:
                    auth_session.output_lines.append(clean)
                    self._refresh_codex_fields(auth_session)
        except Exception as exc:  # pragma: no cover - defensive reader thread
            auth_session.error = str(exc)

    def _refresh_codex_fields(self, auth_session: AuthSession) -> None:
        text = "\n".join(auth_session.output_lines)
        url_match = re.search(r"https://auth\.openai\.com/[^\s]+", text)
        if url_match:
            auth_session.login_url = url_match.group(0)
        code_match = re.search(r"(?:code|one-time code)\D+([A-Z0-9]{4}(?:[-\s]?[A-Z0-9]{4})+)", text, re.IGNORECASE)
        if code_match:
            auth_session.user_code = code_match.group(1).replace(" ", "-")

    def _cleanup_session(self, auth_session: AuthSession) -> None:
        process = auth_session.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        if auth_session.temp_dir is not None:
            shutil.rmtree(auth_session.temp_dir, ignore_errors=True)


def parse_antigravity_callback(value: str, *, expected_state: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Paste the localhost callback URL or authorization code.")
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        query = parse_qs(parsed.query)
        code = str((query.get("code") or [""])[0] or "")
        state = str((query.get("state") or [""])[0] or "")
    else:
        code = text
        state = expected_state
    if not code:
        raise ValueError("The callback did not include an authorization code.")
    if state != expected_state:
        raise ValueError("The callback state did not match this login session.")
    return code, state


def exchange_antigravity_code(
    *,
    code: str,
    code_verifier: str,
    project_id: str = "",
    timeout: int = 20,
) -> dict[str, Any]:
    logger.info(
        "Gemini Antigravity token exchange starting code_present=%s verifier_present=%s project_set=%s oauth=%s",
        bool(str(code or "").strip()),
        bool(str(code_verifier or "").strip()),
        bool(str(project_id or "").strip()),
        antigravity_oauth_config_summary(),
    )
    try:
        client_id, client_secret, redirect_uri = require_antigravity_oauth_config()
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "code_verifier": code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logger.info("Gemini Antigravity token exchange request failed error=%s", type(exc).__name__)
        raise ValueError("Google token exchange could not be reached. Try the Gemini login again.") from exc
    logger.info(
        "Gemini Antigravity token exchange response status=%s content_type=%s",
        token_response.status_code,
        token_response.headers.get("content-type", ""),
    )
    if token_response.status_code >= 400:
        detail = "Google rejected the authorization code."
        try:
            payload = token_response.json()
            error_text = str(payload.get("error_description") or payload.get("error") or "").strip()
            logger.info(
                "Gemini Antigravity token exchange rejected error=%s description=%s",
                str(payload.get("error") or ""),
                error_text[:240],
            )
            if error_text:
                detail = f"{detail} {error_text}"
        except ValueError:
            logger.info("Gemini Antigravity token exchange rejected with non-JSON body")
            pass
        raise ValueError(detail)
    token_payload = token_response.json()
    access_token = str(token_payload.get("access_token") or "")
    refresh_token = str(token_payload.get("refresh_token") or "")
    logger.info(
        "Gemini Antigravity token exchange parsed access_present=%s refresh_present=%s expires_in=%s",
        bool(access_token),
        bool(refresh_token),
        token_payload.get("expires_in"),
    )
    if not access_token or not refresh_token:
        raise ValueError("Google did not return the expected Antigravity tokens. Try the login again.")

    email = ""
    try:
        userinfo = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=timeout,
        )
        if userinfo.status_code < 400:
            email = str((userinfo.json() or {}).get("email") or "")
    except requests.RequestException:
        email = ""

    expires_in = int(token_payload.get("expires_in") or 3600)
    expires_ms = int((time.time() + max(1, expires_in)) * 1000)
    return {
        "google": {
            "type": "oauth",
            "refresh": f"{refresh_token}|{project_id or ''}",
            "access": access_token,
            "expires": expires_ms,
            "email": email,
            "projectId": project_id or "",
        }
    }


def load_json_object(path: Path) -> str:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Auth file did not contain a JSON object.")
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)


def _pkce_verifier() -> str:
    return secrets.token_urlsafe(64)[:96]


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
