from __future__ import annotations

import base64
import json
import logging
import math
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from threading import Lock
from typing import Any, Callable, Iterator, Optional
from urllib.parse import parse_qs, urlparse

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
import requests
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .account_auth import (
    AccountAuthManager,
    AntigravityConfigError,
    antigravity_oauth_config_summary,
    antigravity_redirect_uri,
    exchange_antigravity_code,
    load_json_object,
    parse_antigravity_callback,
)
from .agent.graph import _extract_generation_usage
from .agent.options import AgentRuntimeOptions
from .agent.runtime import OntoPortalAgent
from .antigravity_models import (
    DEFAULT_ANTIGRAVITY_MODEL_REF,
    antigravity_model_options,
    normalize_antigravity_model_ref,
)
from .artifact_store import (
    ArtifactAccessError,
    artifact_expired,
    build_artifact_bundle,
    cleanup_expired_workspaces,
    execution_allows_path,
    file_metadata,
    list_artifact_files,
    read_artifact_diff,
    read_artifact_text,
    resolve_artifact_file,
    sanitize_artifact_path,
)
from .config import get_settings
from .db import EncryptionService, init_db
from .db.base import get_db_session
from .db.models import AssistantMessage
from .db.repositories import (
    create_message,
    create_thread,
    delete_thread,
    ensure_thread,
    get_thread_execution,
    get_latest_thread_execution,
    get_user_settings,
    list_mcp_servers,
    list_thread_messages,
    list_threads,
    replace_mcp_servers,
    update_thread_title,
    upsert_user_settings,
)
from .db.user_context import AssistantUserContext, verify_user_context_headers
from .intent import classify_user_intent
from .mcp_client import McpClient, McpInvocationError
from .opencode_executor import OpenCodeAccountAuth, OpenCodeExecutionResult, OpenCodeExecutor, OpenCodeProviderAuth
from .rag_client import RagClient

logger = logging.getLogger("uvicorn.error").getChild("ontoportal_agent")

app = FastAPI(
    title="OntoPortal Agent API",
    description="Streaming bridge for the MatPortal assistant UI.",
    version="2.0.0",
)

_agent_lock = Lock()
_agent_instance: Optional[OntoPortalAgent] = None
_account_auth_manager = AccountAuthManager()

_LEGACY_DEFAULT_MCP_TIMEOUT_MS = 10_000
_BUILTIN_DEFAULT_MCP_TIMEOUT_MS = 30_000
_MCP_AUTH_NONE = "none"
_MCP_AUTH_API_KEY = "api_key"
_MCP_AUTH_BASIC_USER = "basic_user"
_MCP_AUTH_BASIC_BOT = "basic_bot"
_MCP_AUTH_MODES = {
    _MCP_AUTH_NONE,
    _MCP_AUTH_API_KEY,
    _MCP_AUTH_BASIC_USER,
    _MCP_AUTH_BASIC_BOT,
}
_GOOGLE_OPENAI_BASE_MARKER = "generativelanguage.googleapis.com"
_GOOGLE_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_GOOGLE_GEMINI_API_PROVIDER_ALIASES = {"google_gemini", "gemini", "gemini_api", "google_ai_studio"}
_GOOGLE_GEMINI_FALLBACK_MODELS = (
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
)
_VERTEX_GEMINI_FALLBACK_MODELS = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
_GOOGLE_THOUGHT_REQUEST_MODEL_PREFIXES = (
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
)
_GOOGLE_THOUGHT_STREAM_MODEL_PREFIXES = (
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
)
_MAX_STREAM_ATTEMPTS_PER_MODEL = 2
_STREAM_RETRY_BACKOFF_SECONDS = 1.25
_GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class ChatStreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    thread_id: Optional[str] = None
    thread_title: Optional[str] = None
    mode: Optional[str] = None


class ProviderConfigIn(BaseModel):
    provider: str = "openai_compatible"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    clear_api_key: bool = False


class ProviderCheckIn(ProviderConfigIn):
    scope: str = "generation"


class McpServerIn(BaseModel):
    name: str
    url: str
    auth_mode: str = _MCP_AUTH_API_KEY
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    enabled: bool = True
    timeout_ms: int = _BUILTIN_DEFAULT_MCP_TIMEOUT_MS


class RetrievalSettingsIn(BaseModel):
    chunk_count: int = Field(default=20, ge=1, le=40)


class OpenCodeSettingsIn(BaseModel):
    auth_source: str = "auto"
    auth_kind: Optional[str] = None
    antigravity_model: Optional[str] = None
    auth_json: Optional[str] = None
    codex_auth_json: Optional[str] = None
    clear_account_auth: bool = False


class AssistantSettingsIn(BaseModel):
    generation: ProviderConfigIn = Field(default_factory=ProviderConfigIn)
    embeddings: ProviderConfigIn = Field(default_factory=ProviderConfigIn)
    reranker: ProviderConfigIn = Field(default_factory=lambda: ProviderConfigIn(provider="none"))
    retrieval: RetrievalSettingsIn = Field(default_factory=RetrievalSettingsIn)
    opencode: OpenCodeSettingsIn = Field(default_factory=OpenCodeSettingsIn)
    mcp_servers: list[McpServerIn] = Field(default_factory=list)


class AntigravityAuthStartIn(BaseModel):
    project_id: Optional[str] = None


class AntigravityAuthCompleteIn(BaseModel):
    auth_session_id: str
    callback_url_or_code: str


class ThreadCreateRequest(BaseModel):
    title: Optional[str] = None
    thread_id: Optional[str] = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _iter_text_chunks(text: str, max_chars: int = 220) -> Iterator[str]:
    clean = text.strip()
    if not clean:
        return

    # Prefer sentence-sized chunks; fall back to hard splits for long lines.
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        start = 0
        while start < len(sentence):
            end = min(len(sentence), start + max_chars)
            if end < len(sentence):
                split = sentence.rfind(" ", start, end)
                if split > start:
                    end = split
            chunk = sentence[start:end].strip()
            if chunk:
                yield chunk + (" " if end < len(sentence) else "")
            start = end


def _flatten_chunk_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_flatten_chunk_text(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content"):
            text = _flatten_chunk_text(value.get(key))
            if text:
                return text
        return ""
    return str(value)


def _normalized_chunk_count(value: Any, *, default: int = 20) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(40, parsed))


def _source_document_label(ontology_id: str, version: str) -> str:
    parts = [str(ontology_id or "").strip()]
    if str(version or "").strip():
        parts.append(f"v{str(version).strip()}")
    return " ".join(part for part in parts if part).strip() or "Unknown source"


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def _source_payload_from_rag_chunk(source: Any, index: int) -> dict[str, Any]:
    metadata = dict(getattr(source, "metadata", {}) or {})
    ontology_id = str(getattr(source, "ontology_id", "") or metadata.get("ontology_id") or "").strip()
    version = str(getattr(source, "version", "") or metadata.get("version") or "").strip()
    header = str(metadata.get("header") or metadata.get("section") or "").strip()
    content = str(getattr(source, "content", "") or "").strip()
    rank = metadata.get("rank") or index + 1
    retrieval_score = _safe_float(metadata.get("retrieval_score") or metadata.get("score"))
    rerank_score = _safe_float(metadata.get("rerank_score"))
    return {
        "ontology_id": ontology_id,
        "version": version,
        "document_label": _source_document_label(ontology_id, version),
        "content": content,
        "metadata": {
            "header": header,
            "rank": rank,
            "retrieval_score": retrieval_score,
            "rerank_score": rerank_score,
            "chunk_id": metadata.get("chunk_id") or metadata.get("id") or f"{ontology_id}:{version}:{index + 1}",
        },
    }


def _source_payload_from_mapping(source: Any, index: int) -> dict[str, Any]:
    item = dict(source or {})
    metadata = dict(item.get("metadata") or {})
    ontology_id = str(item.get("ontology_id") or metadata.get("ontology_id") or "unknown").strip()
    version = str(item.get("version") or metadata.get("version") or "unknown").strip()
    header = str(item.get("header") or metadata.get("header") or metadata.get("section") or "").strip()
    content = str(item.get("content") or metadata.get("content") or "").strip()
    rank = metadata.get("rank") or item.get("rank") or index + 1
    retrieval_score = _safe_float(metadata.get("retrieval_score") or metadata.get("score") or item.get("score"))
    rerank_score = _safe_float(metadata.get("rerank_score") or item.get("rerank_score"))
    return {
        "ontology_id": ontology_id,
        "version": version,
        "document_label": str(item.get("document_label") or _source_document_label(ontology_id, version)),
        "content": content,
        "metadata": {
            "header": header,
            "rank": rank,
            "retrieval_score": retrieval_score,
            "rerank_score": rerank_score,
            "chunk_id": metadata.get("chunk_id") or item.get("chunk_id") or f"{ontology_id}:{version}:{index + 1}",
        },
    }


def _compact_user_id(user_id: str | None) -> str:
    text = str(user_id or "").strip()
    if not text:
        return ""
    return text.rstrip("/").rsplit("/", 1)[-1]


def _trim_log_value(value: Any, *, limit: int = 160) -> str:
    text = str(value).replace("\n", "\\n").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _log_event(event: str, **fields: Any) -> str:
    parts = [event]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = json.dumps(_trim_log_value(value), ensure_ascii=False)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def _error_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    match = re.search(r"Error code:\s*(\d+)", str(exc), re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _error_retry_after_seconds(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return max(0, math.ceil(float(retry_after)))
            except ValueError:
                pass

    match = re.search(r"retry in ([0-9.]+)s", str(exc), re.IGNORECASE)
    if not match:
        return None
    try:
        return max(0, math.ceil(float(match.group(1))))
    except ValueError:
        return None


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = _error_status_code(exc)
    if status_code == 429:
        return True
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota exceeded" in text or "rate limit" in text


def _stream_failure_payload(exc: Exception) -> tuple[str, str]:
    if _is_rate_limit_error(exc):
        retry_after = _error_retry_after_seconds(exc)
        if retry_after is not None:
            wait_text = f" Retry in about {retry_after} seconds."
        else:
            wait_text = ""
        return (
            "Provider quota exceeded.",
            "The configured AI provider quota is exhausted."
            f"{wait_text} Add your own API key in AI Settings or try again later.",
        )

    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return (
            "Provider timeout.",
            "The assistant provider timed out while handling the request. Try again or switch the model in AI Settings.",
        )
    if "connection" in text or "temporarily unavailable" in text:
        return (
            "Provider connection failed.",
            "The assistant could not reach the configured provider. Check the provider settings in AI Settings and retry.",
        )
    return ("Assistant request failed.", "Assistant backend failed while handling the request.")


def _uses_google_openai_base(base_url: str | None) -> bool:
    return _GOOGLE_OPENAI_BASE_MARKER in str(base_url or "").strip().lower()


def _uses_gemini_api_provider(provider: str | None) -> bool:
    return str(provider or "").strip().lower() in _GOOGLE_GEMINI_API_PROVIDER_ALIASES


def _uses_vertex_gemini_provider(runtime_options: AgentRuntimeOptions | None) -> bool:
    provider = str(getattr(runtime_options, "generation_provider", "") or "").strip().lower()
    return provider == "vertex_gemini"


def _generation_model_candidates(
    selected_model: str | None,
    *,
    base_url: str | None,
    generation_provider: str | None = None,
) -> list[str]:
    clean_selected = str(selected_model or "").strip()
    candidates: list[str] = []
    if clean_selected:
        candidates.append(clean_selected)

    clean_provider = str(generation_provider or "").strip().lower()
    if clean_provider == "vertex_gemini":
        for candidate in _VERTEX_GEMINI_FALLBACK_MODELS:
            if candidate not in candidates:
                candidates.append(candidate)
    elif _uses_google_openai_base(base_url):
        for candidate in _GOOGLE_GEMINI_FALLBACK_MODELS:
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates or ([clean_selected] if clean_selected else [])


def _can_retry_with_fallback_model(exc: Exception) -> bool:
    status_code = _error_status_code(exc)
    if status_code in {400, 404, 408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    text = str(exc).lower()
    retry_markers = (
        "resource_exhausted",
        "high demand",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection error",
        "unsupported",
        "not found",
        "cannot find field",
        "unknown name",
    )
    return any(marker in text for marker in retry_markers)


def _stream_attempts_for_model(model: str | None, *, base_url: str | None) -> int:
    clean_model = str(model or "").strip()
    if not clean_model:
        return 1
    if _uses_google_openai_base(base_url) and any(
        clean_model.startswith(prefix) for prefix in _GOOGLE_THOUGHT_STREAM_MODEL_PREFIXES
    ):
        return _MAX_STREAM_ATTEMPTS_PER_MODEL
    return 1


def _stream_retry_delay_seconds(attempt_number: int) -> float:
    if attempt_number <= 0:
        return 0.0
    return min(3.0, _STREAM_RETRY_BACKOFF_SECONDS * attempt_number)


@lru_cache(maxsize=4)
def _vertex_service_account_credentials(service_account_json: str):
    payload = json.loads(service_account_json)
    return service_account.Credentials.from_service_account_info(
        payload,
        scopes=[_GOOGLE_CLOUD_PLATFORM_SCOPE],
    )


def _vertex_service_account_project_id(service_account_json: str | None) -> str:
    try:
        payload = json.loads(str(service_account_json or ""))
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("project_id") or "").strip()


def _vertex_access_token(runtime_options: AgentRuntimeOptions | None) -> str:
    settings = get_settings()
    service_account_json = str(
        (getattr(runtime_options, "vertex_service_account_json", None) if runtime_options else None)
        or settings.vertex_service_account_json
        or ""
    ).strip()
    if not service_account_json:
        raise RuntimeError("Vertex Gemini provider requires ONTOAGENT_VERTEX_SERVICE_ACCOUNT_JSON.")

    credentials = _vertex_service_account_credentials(service_account_json)
    credentials.refresh(GoogleAuthRequest())
    return str(credentials.token or "")


def _vertex_endpoint_url(runtime_options: AgentRuntimeOptions | None, model: str) -> str:
    settings = get_settings()
    project = str(
        (getattr(runtime_options, "vertex_project", None) if runtime_options else None)
        or settings.vertex_project
        or ""
    ).strip()
    location = str(
        (getattr(runtime_options, "vertex_location", None) if runtime_options else None)
        or settings.vertex_location
        or "us-central1"
    ).strip()
    if not project:
        raise RuntimeError("Vertex Gemini provider requires ONTOAGENT_VERTEX_PROJECT.")
    clean_model = str(model or "").strip()
    if clean_model.startswith("publishers/google/models/"):
        clean_model = clean_model.split("publishers/google/models/", 1)[1]
    if clean_model.startswith("google/"):
        clean_model = clean_model.split("google/", 1)[1]

    return (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{clean_model}:streamGenerateContent?alt=sse"
    )


def _vertex_openai_base_url(runtime_options: AgentRuntimeOptions | None) -> str:
    settings = get_settings()
    project = str(
        (getattr(runtime_options, "vertex_project", None) if runtime_options else None)
        or settings.vertex_project
        or ""
    ).strip()
    if not project:
        raise RuntimeError("Vertex Gemini provider requires ONTOAGENT_VERTEX_PROJECT.")
    return f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"


def _vertex_openai_model_name(model: str | None) -> str:
    clean_model = str(model or "").strip()
    if clean_model.startswith("publishers/google/models/"):
        clean_model = clean_model.split("publishers/google/models/", 1)[1]
    if clean_model.startswith("google/"):
        return clean_model
    return f"google/{clean_model}" if clean_model else "google/gemini-2.5-pro"


def _vertex_buffered_edit_runtime_options(runtime_options: AgentRuntimeOptions) -> AgentRuntimeOptions:
    settings = get_settings()
    selected_model = str(runtime_options.llm_model or settings.llm_model or "").strip() or "gemini-2.5-pro"
    return AgentRuntimeOptions(
        openai_api_key=_vertex_access_token(runtime_options),
        generation_provider="openai_compatible",
        generation_api_key_configured=True,
        openai_api_base=_vertex_openai_base_url(runtime_options),
        llm_model=_vertex_openai_model_name(selected_model),
        vertex_project=getattr(runtime_options, "vertex_project", None),
        vertex_location=getattr(runtime_options, "vertex_location", None),
        vertex_service_account_json=getattr(runtime_options, "vertex_service_account_json", None),
        rag_top_k=getattr(runtime_options, "rag_top_k", None),
        rag_base_url=getattr(runtime_options, "rag_base_url", None),
        rag_query_path=getattr(runtime_options, "rag_query_path", None),
        mcp_endpoints=getattr(runtime_options, "mcp_endpoints", None),
        mcp_api_key=getattr(runtime_options, "mcp_api_key", None),
        mcp_rag_tool_name=getattr(runtime_options, "mcp_rag_tool_name", None),
    )


def _failure_log_fields(exc: Exception) -> dict[str, Any]:
    return {
        "error_class": exc.__class__.__name__,
        "status_code": _error_status_code(exc),
        "retry_after_seconds": _error_retry_after_seconds(exc),
        "rate_limited": _is_rate_limit_error(exc),
    }


def _google_stream_extra_body(*, base_url: str | None, model: str | None) -> dict[str, Any] | None:
    if not _uses_google_openai_base(base_url):
        return None
    clean_model = str(model or "").strip()
    if not any(clean_model.startswith(prefix) for prefix in _GOOGLE_THOUGHT_REQUEST_MODEL_PREFIXES):
        return None
    return {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinking_level": "low",
                    "include_thoughts": True,
                }
            }
        }
    }


def _strip_google_thought_tags(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("<thought>", "")
    cleaned = cleaned.replace("</thought>", "")
    return cleaned


def _update_usage_from_openai_chunk(usage: dict[str, Any], chunk_payload: dict[str, Any]) -> None:
    usage_payload = chunk_payload.get("usage")
    if isinstance(usage_payload, dict):
        if usage_payload.get("prompt_tokens") is not None:
            usage["prompt_tokens"] = usage_payload.get("prompt_tokens")
        if usage_payload.get("completion_tokens") is not None:
            usage["completion_tokens"] = usage_payload.get("completion_tokens")
        if usage_payload.get("total_tokens") is not None:
            usage["total_tokens"] = usage_payload.get("total_tokens")
    model_name = chunk_payload.get("model")
    if model_name:
        usage["model"] = model_name


def _stream_openai_compatible_events(
    *,
    runtime_options: AgentRuntimeOptions,
    model: str,
    messages: list[Any],
    usage_state: dict[str, Any],
    answer_chunks: list[str],
    reasoning_chunks: list[str],
) -> Iterator[dict[str, Any]]:
    base_url = (runtime_options.openai_api_base if runtime_options else None) or get_settings().openai_api_base
    client = OpenAI(
        api_key=(runtime_options.openai_api_key if runtime_options else "") or get_settings().openai_api_key,
        base_url=base_url,
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system" if isinstance(message, SystemMessage) else "user",
                "content": str(message.content),
            }
            for message in messages
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    extra_body = _google_stream_extra_body(base_url=base_url, model=model)
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    saw_reasoning_text = False
    saw_reasoning_signature = False

    for chunk in client.chat.completions.create(**request_kwargs):
        chunk_payload = chunk.model_dump(mode="json") if hasattr(chunk, "model_dump") else {}
        _update_usage_from_openai_chunk(usage_state, chunk_payload)

        choices = chunk_payload.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        raw_text = _flatten_chunk_text(delta.get("content"))
        if not raw_text:
            google_extra = ((delta.get("extra_content") or {}).get("google") or {})
            if google_extra.get("thought_signature"):
                saw_reasoning_signature = True
            continue

        google_extra = ((delta.get("extra_content") or {}).get("google") or {})
        if google_extra.get("thought_signature"):
            saw_reasoning_signature = True

        if google_extra.get("thought") is True:
            cleaned_thought = _strip_google_thought_tags(raw_text)
            if cleaned_thought:
                saw_reasoning_text = True
                reasoning_chunks.append(cleaned_thought)
                yield {"type": "reasoning_delta", "content": cleaned_thought}
            continue

        cleaned_answer = _strip_google_thought_tags(raw_text)
        if cleaned_answer:
            answer_chunks.append(cleaned_answer)
            yield {"type": "delta", "content": cleaned_answer}

    if saw_reasoning_text:
        usage_state["reasoning_kind"] = "provider_thought_stream"
        usage_state["reasoning_displayable"] = True
    elif saw_reasoning_signature:
        usage_state["reasoning_kind"] = "provider_thought_signature"


def _stream_vertex_gemini_events(
    *,
    runtime_options: AgentRuntimeOptions,
    model: str,
    messages: list[Any],
    usage_state: dict[str, Any],
    answer_chunks: list[str],
    reasoning_chunks: list[str],
) -> Iterator[dict[str, Any]]:
    system_parts: list[dict[str, str]] = []
    user_parts: list[dict[str, str]] = []

    for message in messages:
        text = str(getattr(message, "content", "") or "").strip()
        if not text:
            continue
        if isinstance(message, SystemMessage):
            system_parts.append({"text": text})
        else:
            user_parts.append({"text": text})

    if not user_parts:
        raise RuntimeError("Vertex Gemini request is missing user content.")

    request_body: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": user_parts,
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "includeThoughts": True,
            }
        },
    }
    if system_parts:
        request_body["systemInstruction"] = {"parts": system_parts}

    response = requests.post(
        _vertex_endpoint_url(runtime_options, model),
        headers={
            "Authorization": f"Bearer {_vertex_access_token(runtime_options)}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=request_body,
        stream=True,
        timeout=(15, 300),
    )
    response.raise_for_status()

    saw_reasoning_text = False
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = str(raw_line).strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line.replace("data:", "", 1).strip()
        if not payload:
            continue
        event = json.loads(payload)
        usage_metadata = event.get("usageMetadata") or {}
        if usage_metadata.get("promptTokenCount") is not None:
            usage_state["prompt_tokens"] = usage_metadata.get("promptTokenCount")
        if usage_metadata.get("candidatesTokenCount") is not None:
            usage_state["completion_tokens"] = usage_metadata.get("candidatesTokenCount")
        if usage_metadata.get("totalTokenCount") is not None:
            usage_state["total_tokens"] = usage_metadata.get("totalTokenCount")
        if usage_metadata.get("thoughtsTokenCount") is not None:
            usage_state["reasoning_tokens"] = usage_metadata.get("thoughtsTokenCount")
        if event.get("modelVersion"):
            usage_state["model"] = event.get("modelVersion")

        candidates = event.get("candidates") or []
        if not candidates:
            continue
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        for part in parts:
            text = _flatten_chunk_text(part.get("text"))
            if not text:
                continue
            if part.get("thought") is True:
                saw_reasoning_text = True
                reasoning_chunks.append(text)
                yield {"type": "reasoning_delta", "content": text}
            else:
                answer_chunks.append(text)
                yield {"type": "delta", "content": text}

    if saw_reasoning_text:
        usage_state["reasoning_kind"] = "provider_thought_stream"
        usage_state["reasoning_displayable"] = True


def _usage_allows_reasoning_display(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    if usage.get("reasoning_displayable") is not True:
        return False
    if usage.get("reasoning_kind") != "provider_thought_stream":
        return False
    return True


def _reasoning_is_user_visible(final_state: dict[str, Any]) -> bool:
    usage = final_state.get("generation_usage")
    if not _usage_allows_reasoning_display(usage):
        return False
    return bool(str(final_state.get("generation_reasoning") or "").strip())


def _resolved_default_mcp_urls(settings: Any | None = None) -> set[str]:
    resolved_settings = settings or get_settings()
    endpoints = resolved_settings.default_mcp_endpoints or resolved_settings.resolved_mcp_endpoints()
    return {str(item or "").rstrip("/") for item in endpoints if str(item or "").strip()}


def _normalized_mcp_timeout_ms(
    value: Any,
    *,
    url: str | None = None,
    default: int = _BUILTIN_DEFAULT_MCP_TIMEOUT_MS,
    settings: Any | None = None,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    normalized = max(1000, parsed)
    clean_url = str(url or "").rstrip("/")
    if clean_url and clean_url in _resolved_default_mcp_urls(settings) and normalized == _LEGACY_DEFAULT_MCP_TIMEOUT_MS:
        return _BUILTIN_DEFAULT_MCP_TIMEOUT_MS
    return normalized


def _derive_thread_title(prompt: str, *, max_chars: int = 72) -> str:
    clean = " ".join(str(prompt or "").split())
    if not clean:
        return "New chat"
    if len(clean) <= max_chars:
        return clean.rstrip(" .,:;!-")

    trimmed = clean[:max_chars].rstrip()
    split_at = trimmed.rfind(" ")
    if split_at >= 24:
        trimmed = trimmed[:split_at]
    return trimmed.rstrip(" .,:;!-") + "..."


def _build_chat_model(
    runtime_options: AgentRuntimeOptions | None,
    *,
    model_override: str | None = None,
) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=(runtime_options.openai_api_key if runtime_options else "") or settings.openai_api_key,
        base_url=(runtime_options.openai_api_base if runtime_options else None) or settings.openai_api_base,
        model=model_override or (runtime_options.llm_model if runtime_options else None) or settings.llm_model,
        temperature=0.0,
    )


def _classify_intent(llm: ChatOpenAI, prompt: str) -> str:
    return classify_user_intent(prompt, llm=llm)


def _normalize_chat_mode(mode: str | None) -> str:
    clean = str(mode or "").strip().lower().replace("_", "-")
    if clean in {"edit", "opencode", "open-code", "workspace"}:
        return "edit"
    return "ask"


def _build_response_messages(
    *,
    question: str,
    rag_result: str,
    citations: list[str],
    retrieval_backend: str,
    retrieval_error: str,
) -> list[Any]:
    citation_text = "\n".join(f"- {item}" for item in citations) if citations else "- none"
    return [
        SystemMessage(
            content=(
                "You are the MatPortal ontology assistant.\n"
                "Rules:\n"
                "- Be concise and readable.\n"
                "- Use short paragraphs or bullets only.\n"
                "- Do not use markdown tables unless the user explicitly asks for a table.\n"
                "- Do not include code blocks unless the user asks for code.\n"
                "- Never invent ontology names, class URIs, or facts not present in the retrieved context.\n"
                "- If evidence is weak or missing, state uncertainty clearly.\n"
                "- For recommendation questions, return up to 3 options with a one-line reason each."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\n"
                f"Retrieved context: {rag_result}\n"
                f"Retrieval backend: {retrieval_backend}\n"
                f"Retrieval error: {retrieval_error}\n"
                f"Citations:\n{citation_text}\n"
                "Write a direct answer for the user."
            )
        ),
    ]


def _retrieve_runtime_state(prompt: str, runtime_options: AgentRuntimeOptions | None) -> dict[str, Any]:
    settings = get_settings()
    mcp_endpoints = runtime_options.mcp_endpoints if runtime_options and runtime_options.mcp_endpoints else settings.resolved_mcp_endpoints()
    mcp_api_key = runtime_options.mcp_api_key if runtime_options and runtime_options.mcp_api_key else settings.mcp_api_key
    mcp_rag_tool_name = (
        runtime_options.mcp_rag_tool_name
        if runtime_options and runtime_options.mcp_rag_tool_name
        else settings.mcp_rag_tool_name
    )
    rag_base_url = runtime_options.rag_base_url if runtime_options and runtime_options.rag_base_url else settings.rag_base_url
    rag_query_path = runtime_options.rag_query_path if runtime_options and runtime_options.rag_query_path else settings.rag_query_path
    rag_top_k = _normalized_chunk_count(
        runtime_options.rag_top_k if runtime_options else None,
        default=20,
    )

    state: dict[str, Any] = {
        "citations": [],
        "citation_labels": [],
        "rag_result": "",
        "retrieval_backend": "none",
        "retrieval_error": "",
        "retrieval_chunk_count": rag_top_k,
    }

    if rag_base_url and rag_query_path:
        try:
            result = RagClient(base_url=rag_base_url, query_path=rag_query_path).query(prompt, top_k=rag_top_k)
            state["rag_result"] = result.answer
            state["citations"] = [_source_payload_from_rag_chunk(src, index) for index, src in enumerate(result.sources)]
            state["citation_labels"] = list(dict.fromkeys(item["document_label"] for item in state["citations"]))
            state["retrieval_backend"] = "rag-http"
            return state
        except Exception as err:  # noqa: BLE001 - we intentionally degrade to MCP or non-RAG response.
            state["retrieval_backend"] = "mcp-fallback"
            state["retrieval_error"] = str(err)

    if mcp_endpoints:
        mcp_client = McpClient(mcp_endpoints, api_key=mcp_api_key)
        try:
            mcp_payload = mcp_client.invoke_rag_query(
                prompt,
                tool_name=mcp_rag_tool_name,
                top_k=rag_top_k,
            )
            sources = mcp_payload.get("sources", [])
            state["rag_result"] = str(mcp_payload.get("answer", "") or "")
            state["citations"] = [_source_payload_from_mapping(src, index) for index, src in enumerate(sources)]
            state["citation_labels"] = list(dict.fromkeys(item["document_label"] for item in state["citations"]))
            state["retrieval_backend"] = "mcp"
            return state
        except (McpInvocationError, KeyError, TypeError, ValueError) as err:
            state["retrieval_backend"] = "none"
            existing_error = state.get("retrieval_error")
            state["retrieval_error"] = f"{existing_error}; fallback failed: {err}" if existing_error else str(err)

    state["retrieval_backend"] = "none"
    state["rag_result"] = ""
    state["citations"] = []
    state["citation_labels"] = []

    return state


def _collect_graph_final_state(
    *,
    agent: OntoPortalAgent,
    prompt: str,
    thread_id: str | None,
) -> dict[str, Any]:
    graph_config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    final_state: dict[str, Any] = {}
    updates = (
        agent.graph.stream({"user_input": prompt}, config=graph_config, stream_mode="updates")
        if graph_config
        else agent.graph.stream({"user_input": prompt}, stream_mode="updates")
    )
    for update in updates:
        if not isinstance(update, dict):
            continue
        for node_state in update.values():
            if isinstance(node_state, dict):
                final_state.update(node_state)
    return final_state


def _emit_final_state(final_state: dict[str, Any]) -> Iterator[str]:
    generation_usage = final_state.get("generation_usage")
    if isinstance(generation_usage, dict):
        yield _sse({"type": "usage", "content": generation_usage})

    citations = final_state.get("citations")
    if isinstance(citations, list) and citations:
        yield _sse({"type": "citations", "content": citations})

    sandbox_output = final_state.get("sandbox_output")
    if sandbox_output:
        for line in str(sandbox_output).splitlines():
            clean = line.strip()
            if clean:
                yield _sse({"type": "status", "message": clean})

    final_response = (
        final_state.get("final_response")
        or final_state.get("rag_result")
        or "No response generated."
    )
    for chunk in _iter_text_chunks(str(final_response)):
        yield _sse({"type": "delta", "content": chunk})


def _get_agent() -> OntoPortalAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = OntoPortalAgent()
    return _agent_instance


def _get_encryption_service() -> EncryptionService:
    settings = get_settings()
    return EncryptionService(
        current_key_raw=settings.encryption_key_current,
        previous_key_raw=settings.encryption_key_previous,
    )


def _encryption_required() -> EncryptionService:
    service = _get_encryption_service()
    if not service.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Encryption keys are not configured.",
        )
    return service


def _default_settings_payload() -> dict[str, Any]:
    settings = get_settings()
    mcp_endpoints = settings.default_mcp_endpoints or settings.resolved_mcp_endpoints()
    default_mcp_auth_mode = _normalize_mcp_auth_mode(
        getattr(settings, "default_mcp_auth_mode", _MCP_AUTH_NONE),
        api_key=str(getattr(settings, "default_mcp_api_key", "") or ""),
    )
    default_generation_provider = settings.default_generation_provider
    default_generation_base = (
        settings.default_generation_base_url
        if settings.default_generation_base_url is not None
        else (
            ""
            if default_generation_provider == "vertex_gemini"
            else (_GOOGLE_GEMINI_OPENAI_BASE_URL if _uses_gemini_api_provider(default_generation_provider) else settings.openai_api_base or "")
        )
    )
    return {
        "generation": {
            "provider": default_generation_provider,
            "model": settings.default_generation_model or settings.llm_model,
            "api_key": "",
            "base_url": default_generation_base,
        },
        "embeddings": {
            "provider": settings.default_embeddings_provider,
            "model": settings.default_embeddings_model,
            "api_key": "",
            "base_url": settings.default_embeddings_base_url or settings.openai_api_base or "",
        },
        "reranker": {
            "provider": settings.default_reranker_provider,
            "model": settings.default_reranker_model,
            "api_key": "",
            "base_url": settings.default_reranker_base_url or "",
        },
        "retrieval": {
            "chunk_count": 20,
        },
        "opencode": {
            "auth_source": "auto",
            "auth_kind": "",
            "auth_json": "",
            "codex_auth_json": "",
        },
        "mcp_servers": [
            {
                "name": f"MCP {index + 1}",
                "url": endpoint,
                "auth_mode": default_mcp_auth_mode,
                "username": "",
                "password": "",
                "api_key": "",
                "enabled": True,
                "timeout_ms": _BUILTIN_DEFAULT_MCP_TIMEOUT_MS,
            }
            for index, endpoint in enumerate(mcp_endpoints)
        ],
    }


def _normalize_provider(provider_payload: dict[str, Any], default_payload: dict[str, Any]) -> dict[str, Any]:
    api_key = str(provider_payload.get("api_key", "") or "")
    if api_key == "__configured__":
        api_key = ""
    normalized = dict(default_payload)
    normalized.update(
        {
            "provider": str(provider_payload.get("provider", default_payload.get("provider", "openai_compatible"))),
            "model": str(provider_payload.get("model", default_payload.get("model", "")) or ""),
            "api_key": api_key,
            "base_url": str(provider_payload.get("base_url", default_payload.get("base_url", "")) or ""),
        }
    )
    return normalized


def _sanitize_provider_error(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "Provider check failed."
    text = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[redacted]", text)
    text = re.sub(r"sk-[0-9A-Za-z_-]{12,}", "[redacted]", text)
    text = re.sub(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\"'\s,}]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s\"']+", r"\1[redacted]", text)
    return text[:240]


def _provider_check_url(base_url: str) -> str:
    clean = str(base_url or "").strip().rstrip("/")
    if not clean:
        clean = "https://api.openai.com/v1"
    return f"{clean}/models"


def _check_openai_compatible_provider(*, api_key: str, base_url: str, model: str | None) -> dict[str, Any]:
    if not api_key.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="API key is required.")
    response = requests.get(
        _provider_check_url(base_url),
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        timeout=20,
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_sanitize_provider_error(response.text or f"Provider returned HTTP {response.status_code}."),
        )
    body = response.json() if response.content else {}
    model_ids = {
        str(item.get("id") or item.get("name") or "")
        for item in body.get("data", [])
        if isinstance(item, dict)
    }
    requested_model = str(model or "").strip()
    return {
        "ok": True,
        "model_available": bool(requested_model and requested_model in model_ids) if model_ids else None,
        "models_seen": len(model_ids),
    }


def _has_persisted_secret(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text != "__configured__"


def _normalize_mcp_auth_mode(
    value: Any,
    *,
    api_key: str = "",
    username: str = "",
    password: str = "",
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in _MCP_AUTH_MODES:
        if str(api_key or "").strip():
            return _MCP_AUTH_API_KEY
        if str(username or "").strip() or str(password or "").strip():
            return _MCP_AUTH_BASIC_USER
        return _MCP_AUTH_NONE
    return normalized


def _basic_auth_header_value(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _normalize_opencode_settings(raw_payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_payload or {}
    auth_source = str(raw.get("auth_source", "auto") or "auto").strip().lower()
    if auth_source not in {"auto", "generation_key", "opencode_builtin", "account_auth"}:
        auth_source = "auto"
    auth_kind = str(raw.get("auth_kind", "") or "").strip().lower()
    if auth_kind not in {"", "gemini_antigravity", "codex"}:
        auth_kind = ""
    antigravity_default = str(getattr(get_settings(), "opencode_antigravity_model", "") or "").strip() or DEFAULT_ANTIGRAVITY_MODEL_REF
    return {
        "auth_source": auth_source,
        "auth_kind": auth_kind,
        "antigravity_model": normalize_antigravity_model_ref(raw.get("antigravity_model"), default=antigravity_default),
        "auth_json": str(raw.get("auth_json", "") or ""),
        "codex_auth_json": str(raw.get("codex_auth_json", "") or ""),
    }


def _validate_auth_json(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{label} must be valid JSON.",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{label} must be a JSON object.",
        )
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)


def _normalize_settings_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    defaults = _default_settings_payload()
    generation = _normalize_provider(raw_payload.get("generation", {}), defaults["generation"])
    if _uses_gemini_api_provider(generation.get("provider")) and not str(generation.get("base_url") or "").strip():
        generation["base_url"] = _GOOGLE_GEMINI_OPENAI_BASE_URL
    embeddings = _normalize_provider(raw_payload.get("embeddings", {}), defaults["embeddings"])
    reranker = _normalize_provider(raw_payload.get("reranker", {}), defaults["reranker"])
    retrieval = {
        "chunk_count": _normalized_chunk_count(
            raw_payload.get("retrieval", {}).get("chunk_count"),
            default=int(defaults["retrieval"]["chunk_count"]),
        )
    }

    raw_mcp_servers = raw_payload.get("mcp_servers")
    if raw_mcp_servers is None:
        raw_mcp_servers = defaults.get("mcp_servers", [])

    mcp_servers = []
    for item in raw_mcp_servers:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        raw_api_key = str(item.get("api_key", "") or "")
        raw_username = str(item.get("username", "") or "")
        raw_password = str(item.get("password", "") or "")
        api_key = raw_api_key
        username = raw_username
        password = raw_password
        if raw_api_key == "__configured__":
            api_key = ""
        if raw_username == "__configured__":
            username = ""
        if raw_password == "__configured__":
            password = ""
        auth_mode = _normalize_mcp_auth_mode(
            item.get("auth_mode"),
            api_key=raw_api_key,
            username=raw_username,
            password=raw_password,
        )
        if auth_mode != _MCP_AUTH_API_KEY:
            api_key = ""
        if auth_mode != _MCP_AUTH_BASIC_USER:
            username = ""
            password = ""
        mcp_servers.append(
            {
                "name": str(item.get("name", "MCP")).strip() or "MCP",
                "url": url,
                "auth_mode": auth_mode,
                "username": username,
                "password": password,
                "api_key": api_key,
                "enabled": bool(item.get("enabled", True)),
                "timeout_ms": _normalized_mcp_timeout_ms(item.get("timeout_ms"), url=url),
            }
        )

    return {
        "generation": generation,
        "embeddings": embeddings,
        "reranker": reranker,
        "retrieval": retrieval,
        "opencode": _normalize_opencode_settings(raw_payload.get("opencode", defaults.get("opencode", {}))),
        "mcp_servers": mcp_servers,
    }


def _redact_provider(provider_payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(provider_payload)
    has_secret = bool((provider_payload.get("api_key") or "").strip())
    redacted["api_key"] = "__configured__" if has_secret else ""
    return redacted


def _redact_mcp_server(item: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(item)
    auth_mode = _normalize_mcp_auth_mode(
        item.get("auth_mode"),
        api_key=str(item.get("api_key", "") or ""),
        username=str(item.get("username", "") or ""),
        password=str(item.get("password", "") or ""),
    )
    redacted["auth_mode"] = auth_mode
    has_api_key = bool((item.get("api_key") or "").strip())
    has_password = bool((item.get("password") or "").strip())
    redacted["api_key"] = "__configured__" if auth_mode == _MCP_AUTH_API_KEY and has_api_key else ""
    redacted["username"] = str(item.get("username") or "").strip() if auth_mode == _MCP_AUTH_BASIC_USER else ""
    redacted["password"] = "__configured__" if auth_mode == _MCP_AUTH_BASIC_USER and has_password else ""
    return redacted


def _serialize_settings_for_output(payload: dict[str, Any]) -> dict[str, Any]:
    opencode = _normalize_opencode_settings(payload.get("opencode", {}))
    opencode["auth_json"] = "__configured__" if opencode.get("auth_json") else ""
    opencode["codex_auth_json"] = "__configured__" if opencode.get("codex_auth_json") else ""
    return {
        "generation": _redact_provider(payload["generation"]),
        "embeddings": _redact_provider(payload["embeddings"]),
        "reranker": _redact_provider(payload["reranker"]),
        "retrieval": payload.get("retrieval", {"chunk_count": 20}),
        "opencode": opencode,
        "mcp_servers": [_redact_mcp_server(item) for item in payload.get("mcp_servers", [])],
    }


def _assistant_installed_skills(settings_payload: dict[str, Any]) -> list[dict[str, Any]]:
    settings = get_settings()
    opencode = _normalize_opencode_settings(settings_payload.get("opencode", {}))
    antigravity_connected = (
        opencode.get("auth_source") == "account_auth"
        and opencode.get("auth_kind") == "gemini_antigravity"
        and _has_persisted_secret(opencode.get("auth_json"))
    )
    antigravity_model = normalize_antigravity_model_ref(
        opencode.get("antigravity_model"),
        default=str(settings.opencode_antigravity_model or "") or DEFAULT_ANTIGRAVITY_MODEL_REF,
    )
    skills = [
        {
            "id": "ontology_edit_workflow",
            "name": "Ontology edit workflow",
            "category": "OpenCode",
            "enabled": True,
            "status": "installed",
            "description": "Plans ontology edits, inspects RAG/API context, drafts proposal artifacts, and prepares operator review notes.",
            "details": [
                "Requires RAG retrieval before drafting.",
                "Requires exact OntoPortal/API inspection before and after drafting.",
                "Writes proposal TTL/RDF/OWL, operator notes, and draft submission files into an isolated workspace.",
            ],
        },
        {
            "id": "matportal_rag_mcp",
            "name": "MatPortal RAG MCP",
            "category": "MCP",
            "enabled": True,
            "status": "installed",
            "description": "Semantic retrieval for source chunks, terminology discovery, and provenance.",
            "details": [
                f"Tool/server name: {settings.opencode_rag_mcp_name}",
                f"Endpoint: {str(settings.opencode_rag_mcp_url or '').strip() or settings.rag_base_url.rstrip('/') + '/mcp'}",
                f"Timeout: {settings.opencode_rag_mcp_timeout_ms} ms",
            ],
        },
        {
            "id": "ontoportal_api_mcp",
            "name": "OntoPortal API MCP",
            "category": "MCP",
            "enabled": True,
            "status": "installed",
            "description": "Exact portal/API access for ontology metadata, terms, submissions, and full ontology inspection.",
            "details": [
                f"Tool/server name: {settings.opencode_mcp_name}",
                f"Mode: {settings.opencode_mcp_mode}",
                f"Timeout: {settings.opencode_mcp_timeout_ms} ms",
            ],
        },
        {
            "id": "robot_validation",
            "name": "ROBOT validation",
            "category": "Validation",
            "enabled": bool(settings.opencode_robot_enabled),
            "status": "enabled" if settings.opencode_robot_enabled else "disabled",
            "description": "Runs RDF parse checks plus ROBOT verify/report for ontology artifacts when configured.",
            "details": [
                f"Java: {settings.opencode_robot_java_path}",
                f"ROBOT jar: {settings.opencode_robot_jar_path or 'auto/runtime'}",
            ],
        },
        {
            "id": "artifact_review",
            "name": "Artifact review and downloads",
            "category": "Artifacts",
            "enabled": True,
            "status": "installed",
            "description": "Lists generated files, shows diffs/content, and provides individual or bundle downloads.",
            "details": [
                f"Retention: {settings.opencode_artifact_retention_days} day(s)",
                f"Max diff: {settings.opencode_max_diff_chars} characters",
            ],
        },
        {
            "id": "antigravity_search",
            "name": "Antigravity Google search",
            "category": "Research",
            "enabled": antigravity_connected,
            "status": "connected" if antigravity_connected else "available after Gemini login",
            "description": "Lets OpenCode use provider-native Google search for domain research when the user connects Gemini Antigravity.",
            "details": [
                f"Plugin: {settings.opencode_antigravity_plugin}",
                f"Selected model: {antigravity_model}",
            ],
        },
        {
            "id": "exa_websearch",
            "name": "OpenCode Exa web search",
            "category": "Research",
            "enabled": bool(settings.opencode_exa_websearch_enabled),
            "status": "enabled" if settings.opencode_exa_websearch_enabled else "disabled",
            "description": "Optional OpenCode websearch/webfetch permission path for Exa-backed research.",
            "details": [
                "Use Antigravity search first when Gemini Antigravity account auth is connected.",
            ],
        },
        {
            "id": "opencode_release_guards",
            "name": "OpenCode release guards",
            "category": "Safety",
            "enabled": True,
            "status": "guarded",
            "description": "Feature flags keep interactive sessions, publishing, Mobi handoff, and strict workflow gates explicit during rollout.",
            "details": [
                f"Interactive sessions: {'enabled' if settings.opencode_interactive_sessions_enabled else 'disabled'}",
                f"Strict workflow gates: {'enabled' if settings.opencode_strict_workflow_enabled else 'disabled'}",
                f"Apply/publish actions: {'enabled' if settings.opencode_apply_publish_enabled else 'disabled'}",
                f"Mobi handoff: {'enabled' if settings.opencode_mobi_handoff_enabled else 'disabled'}",
                f"Strict sandbox: {'enabled' if settings.opencode_strict_sandbox_enabled else 'disabled'}",
                f"Concurrency: {settings.opencode_user_concurrency_limit} per user / {settings.opencode_global_concurrency_limit} global",
            ],
        },
    ]
    if bool(settings.opencode_hybrid_ask_enabled):
        skills.append(
            {
                "id": "hybrid_ask",
                "name": "Hybrid Ask generation",
                "category": "Ask",
                "enabled": True,
                "status": "enabled",
                "description": "Retrieves chunks in MatPortal, then lets OpenCode generate the Ask answer from backend-owned context.",
                "details": ["Retrieved chunks remain MatPortal-owned and visible in the chat."],
            }
        )
    return skills


def _persist_opencode_account_auth(
    session: Session,
    *,
    user_id: str,
    auth_kind: str,
    opencode_auth_json: str | None = None,
    codex_auth_json: str | None = None,
) -> dict[str, Any]:
    encryption_service = _encryption_required()
    settings_payload = _load_effective_settings(session, user_id=user_id, include_secrets=True)
    opencode = _normalize_opencode_settings(settings_payload.get("opencode", {}))
    opencode["auth_source"] = "account_auth"
    opencode["auth_kind"] = auth_kind
    if opencode_auth_json is not None:
        opencode["auth_json"] = _validate_auth_json(opencode_auth_json, label="OpenCode auth JSON")
    if codex_auth_json is not None:
        opencode["codex_auth_json"] = _validate_auth_json(codex_auth_json, label="Codex auth JSON")
    settings_payload["opencode"] = opencode
    settings_blob = {
        "generation": settings_payload["generation"],
        "embeddings": settings_payload["embeddings"],
        "reranker": settings_payload["reranker"],
        "retrieval": settings_payload["retrieval"],
        "opencode": settings_payload["opencode"],
    }
    encrypted_payload, key_version = encryption_service.encrypt_json(settings_blob)
    upsert_user_settings(
        session,
        user_id=user_id,
        settings_encrypted=encrypted_payload,
        key_version=key_version,
    )
    return _serialize_settings_for_output(settings_payload)


def _load_effective_settings(
    session: Session,
    *,
    user_id: str,
    include_secrets: bool,
) -> dict[str, Any]:
    payload = _default_settings_payload()
    encryption_service = _get_encryption_service()

    row = get_user_settings(session, user_id=user_id)
    if row is not None and encryption_service.enabled:
        try:
            decrypted, _ = encryption_service.decrypt_json(row.settings_encrypted)
            payload = _normalize_settings_payload(decrypted)
        except Exception as exc:
            logger.warning("Failed to decrypt user settings for %s: %s", user_id, exc)

    mcp_rows = list_mcp_servers(session, user_id=user_id)
    if mcp_rows:
        resolved = []
        for server in mcp_rows:
            auth_mode = _normalize_mcp_auth_mode(getattr(server, "auth_mode", None))
            api_key = ""
            password = ""
            if include_secrets and server.api_key_encrypted and encryption_service.enabled:
                try:
                    api_key_payload, _ = encryption_service.decrypt_json(server.api_key_encrypted)
                    api_key = str(api_key_payload.get("api_key", "") or "")
                except Exception as exc:
                    logger.warning("Failed to decrypt MCP server key for %s/%s: %s", user_id, server.id, exc)
            elif not include_secrets and server.api_key_encrypted:
                api_key = "__configured__"
            if include_secrets and getattr(server, "password_encrypted", None) and encryption_service.enabled:
                try:
                    password_payload, _ = encryption_service.decrypt_json(server.password_encrypted)
                    password = str(password_payload.get("password", "") or "")
                except Exception as exc:
                    logger.warning("Failed to decrypt MCP server password for %s/%s: %s", user_id, server.id, exc)
            elif not include_secrets and getattr(server, "password_encrypted", None):
                password = "__configured__"

            resolved.append(
                {
                    "name": server.name,
                    "url": server.url,
                    "auth_mode": auth_mode,
                    "username": str(getattr(server, "username", "") or ""),
                    "password": password,
                    "api_key": api_key,
                    "enabled": bool(server.enabled),
                    "timeout_ms": _normalized_mcp_timeout_ms(server.timeout_ms, url=server.url),
                }
            )
        payload["mcp_servers"] = resolved

    return payload


def _runtime_mcp_headers_for_server(item: dict[str, Any], settings: Any) -> dict[str, str]:
    auth_mode = _normalize_mcp_auth_mode(
        item.get("auth_mode"),
        api_key=str(item.get("api_key", "") or ""),
        username=str(item.get("username", "") or ""),
        password=str(item.get("password", "") or ""),
    )
    if auth_mode == _MCP_AUTH_API_KEY:
        api_key = str(item.get("api_key", "") or "").strip()
        return {"X-API-Key": api_key} if api_key else {}
    if auth_mode == _MCP_AUTH_BASIC_USER:
        username = str(item.get("username", "") or "").strip()
        password = str(item.get("password", "") or "").strip()
        if username and password:
            return {"Authorization": _basic_auth_header_value(username, password)}
        return {}
    if auth_mode == _MCP_AUTH_BASIC_BOT:
        username = str(getattr(settings, "mcp_bot_username", "") or "").strip()
        password = str(getattr(settings, "mcp_bot_password", "") or "").strip()
        if username and password:
            return {"Authorization": _basic_auth_header_value(username, password)}
        return {}
    return {}


def _runtime_options_from_settings(settings_payload: dict[str, Any]) -> AgentRuntimeOptions:
    settings = get_settings()
    generation = settings_payload.get("generation", {})
    retrieval = settings_payload.get("retrieval", {})
    opencode = _normalize_opencode_settings(settings_payload.get("opencode", {}))
    mcp_servers = settings_payload.get("mcp_servers", [])
    enabled_mcp = [item for item in mcp_servers if bool(item.get("enabled", True))]
    mcp_endpoint_configs: list[dict[str, Any]] = []
    for item in enabled_mcp:
        name = str(item.get("name", "")).strip()
        endpoint = str(item.get("url", "")).strip()
        if not endpoint:
            continue
        headers = _runtime_mcp_headers_for_server(item, settings)
        mcp_endpoint_configs.append(
            {
                "name": name or None,
                "url": endpoint,
                "headers": headers or None,
                "timeout_ms": _normalized_mcp_timeout_ms(item.get("timeout_ms"), url=endpoint),
            }
        )

    generation_api_key = str(generation.get("api_key") or "").strip()
    resolved_llm_model = str(generation.get("model") or "").strip() or settings.llm_model
    resolved_generation_provider = (
        str(generation.get("provider") or "").strip() or settings.default_generation_provider
    )
    clean_generation_provider = resolved_generation_provider.strip().lower()
    if clean_generation_provider == "vertex_gemini":
        resolved_vertex_service_account_json = generation_api_key or getattr(settings, "vertex_service_account_json", None)
        resolved_vertex_project = (
            str(generation.get("project") or "").strip()
            or _vertex_service_account_project_id(generation_api_key)
            or str(getattr(settings, "vertex_project", "") or "").strip()
        )
        resolved_vertex_location = (
            str(generation.get("location") or "").strip()
            or str(getattr(settings, "vertex_location", "") or "").strip()
            or "us-central1"
        )
        resolved_openai_key = settings.openai_api_key
        resolved_openai_base = str(generation.get("base_url") or "").strip() or settings.openai_api_base
    elif _uses_gemini_api_provider(clean_generation_provider):
        resolved_vertex_service_account_json = getattr(settings, "vertex_service_account_json", None)
        resolved_vertex_project = getattr(settings, "vertex_project", None)
        resolved_vertex_location = getattr(settings, "vertex_location", "us-central1")
        resolved_generation_provider = "openai_compatible"
        resolved_openai_key = generation_api_key or settings.openai_api_key
        resolved_openai_base = str(generation.get("base_url") or "").strip() or _GOOGLE_GEMINI_OPENAI_BASE_URL
    else:
        resolved_vertex_service_account_json = getattr(settings, "vertex_service_account_json", None)
        resolved_vertex_project = getattr(settings, "vertex_project", None)
        resolved_vertex_location = getattr(settings, "vertex_location", "us-central1")
        resolved_openai_key = generation_api_key or settings.openai_api_key
        resolved_openai_base = str(generation.get("base_url") or "").strip() or settings.openai_api_base
    resolved_mcp_endpoints = mcp_endpoint_configs or settings.default_mcp_endpoints or settings.resolved_mcp_endpoints()

    return AgentRuntimeOptions(
        openai_api_key=resolved_openai_key,
        generation_provider=resolved_generation_provider,
        generation_api_key_configured=bool(generation_api_key),
        openai_api_base=resolved_openai_base,
        llm_model=resolved_llm_model,
        vertex_project=resolved_vertex_project,
        vertex_location=resolved_vertex_location,
        vertex_service_account_json=resolved_vertex_service_account_json,
        rag_top_k=_normalized_chunk_count(retrieval.get("chunk_count"), default=20),
        rag_base_url=settings.rag_base_url,
        rag_query_path=settings.rag_query_path,
        mcp_endpoints=resolved_mcp_endpoints,
        mcp_api_key=settings.default_mcp_api_key or settings.mcp_api_key,
        mcp_rag_tool_name=settings.mcp_rag_tool_name,
        opencode_auth_source=opencode["auth_source"],
        opencode_auth_kind=opencode["auth_kind"],
        opencode_antigravity_model=opencode["antigravity_model"],
        opencode_auth_json=opencode["auth_json"],
        codex_auth_json=opencode["codex_auth_json"],
    )


def _cleanup_old_history(session: Session, *, user_id: str) -> None:
    retention_days = max(1, int(get_settings().history_retention_days))
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    session.execute(
        delete(AssistantMessage).where(
            AssistantMessage.user_id == user_id,
            AssistantMessage.created_at < cutoff,
        )
    )
    session.commit()


def _resolve_user_context(request: Request) -> AssistantUserContext:
    settings = get_settings()
    try:
        return verify_user_context_headers(
            request.headers,
            secret=settings.user_context_secret,
            ttl_seconds=settings.user_context_ttl_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


def _serialize_thread(thread) -> dict[str, Any]:
    return {
        "thread_id": thread.thread_id,
        "title": thread.title or "",
        "created_at": thread.created_at.isoformat() if thread.created_at else None,
        "updated_at": thread.updated_at.isoformat() if thread.updated_at else None,
    }


def _serialize_message(message) -> dict[str, Any]:
    usage = message.usage_json or {}
    reasoning_summary = message.reasoning_summary or ""
    if not _usage_allows_reasoning_display(usage):
        reasoning_summary = ""
    return {
        "id": message.id,
        "thread_id": message.thread_id,
        "role": message.role,
        "content": message.content,
        "reasoning_summary": reasoning_summary,
        "usage": usage,
        "citations": message.citations_json or [],
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def _persistable_reasoning_summary(final_state: dict[str, Any]) -> str:
    if not _reasoning_is_user_visible(final_state):
        return ""
    return str(final_state.get("generation_reasoning") or "").strip()


def _opencode_provider_auth_from_runtime_options(
    runtime_options: AgentRuntimeOptions | None,
) -> OpenCodeProviderAuth | None:
    if runtime_options is None:
        return None
    auth_source = str(getattr(runtime_options, "opencode_auth_source", "auto") or "auto").strip().lower()
    if auth_source in {"opencode_builtin", "account_auth"}:
        return None
    if not bool(getattr(runtime_options, "generation_api_key_configured", False)):
        return None

    model = str(getattr(runtime_options, "llm_model", "") or "").strip()
    if not model:
        return None

    provider = str(getattr(runtime_options, "generation_provider", "") or "").strip().lower()
    if provider == "vertex_gemini":
        if not str(getattr(runtime_options, "vertex_service_account_json", "") or "").strip():
            return None
        buffered_runtime_options = _vertex_buffered_edit_runtime_options(runtime_options)
        api_key = str(getattr(buffered_runtime_options, "openai_api_key", "") or "").strip()
        base_url = str(getattr(buffered_runtime_options, "openai_api_base", "") or "").strip()
        vertex_model = str(getattr(buffered_runtime_options, "llm_model", "") or "").strip()
        if not api_key or not base_url or not vertex_model:
            return None
        return OpenCodeProviderAuth(
            provider_id="matportal-user",
            model=vertex_model,
            api_key=api_key,
            base_url=base_url,
            name="MatPortal user Vertex AI account",
        )

    api_key = str(getattr(runtime_options, "openai_api_key", "") or "").strip()
    if not api_key:
        return None

    settings = get_settings()
    base_url = (
        str(getattr(runtime_options, "openai_api_base", "") or "").strip()
        or str(settings.openai_api_base or "").strip()
        or "https://api.openai.com/v1"
    )
    return OpenCodeProviderAuth(
        provider_id="matportal-user",
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def _opencode_auth_source_from_runtime_options(runtime_options: AgentRuntimeOptions | None) -> str:
    auth_source = str(getattr(runtime_options, "opencode_auth_source", "auto") or "auto").strip().lower()
    if auth_source not in {"auto", "generation_key", "opencode_builtin", "account_auth"}:
        return "auto"
    return auth_source


def _opencode_account_auth_from_runtime_options(
    runtime_options: AgentRuntimeOptions | None,
) -> OpenCodeAccountAuth | None:
    if runtime_options is None:
        return None
    auth_source = _opencode_auth_source_from_runtime_options(runtime_options)
    if auth_source != "account_auth":
        return None
    opencode_auth_json = str(getattr(runtime_options, "opencode_auth_json", "") or "").strip()
    codex_auth_json = str(getattr(runtime_options, "codex_auth_json", "") or "").strip()
    if not opencode_auth_json and not codex_auth_json:
        return None
    return OpenCodeAccountAuth(
        kind=str(getattr(runtime_options, "opencode_auth_kind", "") or "").strip() or "account_auth",
        opencode_auth_json=opencode_auth_json or None,
        codex_auth_json=codex_auth_json or None,
        model_ref=str(getattr(runtime_options, "opencode_antigravity_model", "") or "").strip() or None,
    )


def _stream_opencode_execution(
    *,
    prompt: str,
    thread_id: str | None,
    trace_id: str,
    runtime_options: AgentRuntimeOptions | None = None,
    resume_workspace: str | None = None,
    resume_session_id: str | None = None,
) -> Iterator[str]:
    executor = OpenCodeExecutor(
        provider_auth=_opencode_provider_auth_from_runtime_options(runtime_options),
        account_auth=_opencode_account_auth_from_runtime_options(runtime_options),
        mcp_servers=runtime_options.mcp_endpoints if runtime_options else None,
    )
    stream = executor.stream(
        prompt=prompt,
        thread_id=thread_id,
        trace_id=trace_id,
        resume_workspace=resume_workspace,
        resume_session_id=resume_session_id,
    )
    while True:
        try:
            event = next(stream)
        except StopIteration as stop:
            return stop.value
        yield _sse(event)


def _opencode_success_response(result: OpenCodeExecutionResult) -> str:
    verification_url = _extract_antigravity_verification_url(result.console_lines)
    if verification_url:
        return (
            "Gemini Antigravity needs account verification before it can continue. "
            f"Use this link, then retry:\n{verification_url}"
        )
    if _contains_antigravity_verification_error(result.console_lines):
        return "Gemini Antigravity needs account verification before it can continue. Complete verification and retry."
    provider_error = _antigravity_provider_error_message(result.console_lines)
    if provider_error:
        return provider_error
    summary = str(result.final_text or "").strip()
    if summary:
        return summary

    if result.changed_files:
        return (
            f"OpenCode prepared an ontology edit proposal in `{result.workspace}` "
            f"touching {len(result.changed_files)} file(s)."
        )

    return f"OpenCode finished in `{result.workspace}` without writing any files."


def _opencode_failure_response(result: OpenCodeExecutionResult) -> str:
    if result.blocked:
        detail = str(result.blocked_reason or "unsafe command detected").strip()
        return f"OpenCode run was stopped by sandbox policy: {detail}."
    verification_url = _extract_antigravity_verification_url(result.console_lines)
    if verification_url:
        return (
            "Gemini Antigravity blocked this run pending Google account verification. "
            f"Open this link and complete verification, then retry:\n{verification_url}"
        )
    if _contains_antigravity_verification_error(result.console_lines):
        return "Gemini Antigravity blocked this run pending Google account verification. Complete verification and retry."
    provider_error = _antigravity_provider_error_message(result.console_lines)
    if provider_error:
        return provider_error
    if result.changed_files:
        return (
            f"OpenCode did not finish cleanly, but it still changed {len(result.changed_files)} file(s) "
            f"in `{result.workspace}` for review."
        )
    return "OpenCode could not finish the ontology edit proposal."


def _contains_antigravity_verification_error(lines: list[str]) -> bool:
    for line in lines:
        text = str(line or "")
        if "Verify your account to continue" in text:
            return True
    return False


def _extract_antigravity_verification_url(lines: list[str]) -> str:
    for line in lines:
        text = str(line or "")
        if "accounts.google.com/signin/continue" not in text:
            continue
        match = re.search(r"https://accounts\.google\.com/signin/continue[^\s\"\\]+", text)
        if match:
            return match.group(0)
    return ""


def _joined_console_text(lines: list[str]) -> str:
    return "\n".join(str(line or "") for line in lines)


def _extract_antigravity_project_id(lines: list[str]) -> str:
    match = re.search(r"projects/([a-z][a-z0-9-]{4,}[a-z0-9])", _joined_console_text(lines), re.IGNORECASE)
    return match.group(1) if match else ""


def _extract_antigravity_requested_model(lines: list[str]) -> str:
    text = _joined_console_text(lines)
    match = re.search(r"Requested Model:\s*([^\n]+)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"Effective Model:\s*([^\n]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _contains_antigravity_iam_error(lines: list[str]) -> bool:
    text = _joined_console_text(lines)
    return "cloudaicompanion.companions.generateChat" in text or "IAM_PERMISSION_DENIED" in text


def _contains_antigravity_tool_schema_error(lines: list[str]) -> bool:
    text = _joined_console_text(lines)
    return bool(
        re.search(
            r"custom\.input_schema|input_schema\.properties|Property keys should match pattern|INVALID_ARGUMENT",
            text,
            re.IGNORECASE,
        )
    )


def _contains_antigravity_provider_error(lines: list[str]) -> bool:
    return bool(
        _extract_antigravity_verification_url(lines)
        or _contains_antigravity_verification_error(lines)
        or _contains_antigravity_iam_error(lines)
        or _contains_antigravity_tool_schema_error(lines)
    )


def _antigravity_provider_error_message(lines: list[str]) -> str:
    if _contains_antigravity_iam_error(lines):
        project_id = _extract_antigravity_project_id(lines)
        project_label = f"project `{project_id}`" if project_id else "the configured Google Cloud project"
        return (
            "Gemini Antigravity reached Google, but the connected account is missing Gemini Code Assist "
            f"permission for {project_label} (`cloudaicompanion.companions.generateChat`). "
            "Ask an admin to grant access or reconnect a Google account with access, then retry."
        )

    if _contains_antigravity_tool_schema_error(lines):
        model = _extract_antigravity_requested_model(lines) or "the selected Antigravity model"
        return (
            f"{model} rejected one of the MatPortal tool definitions before the run started. "
            "The assistant runtime needs the provider-safe OntoPortal MCP schema, then this request can be retried."
        )

    return ""


def _opencode_usage_payload(
    result: OpenCodeExecutionResult,
    runtime_options: AgentRuntimeOptions | None = None,
) -> dict[str, Any]:
    model = str(result.model or get_settings().opencode_model or "opencode")
    execution = result.execution_payload()
    verification_url = _extract_antigravity_verification_url(result.console_lines)
    if verification_url:
        execution["verification_url"] = verification_url
        execution["verification_required"] = True
    elif _contains_antigravity_verification_error(result.console_lines):
        execution["verification_required"] = True
    if _contains_antigravity_iam_error(result.console_lines):
        execution["provider_error"] = "antigravity_iam_permission_denied"
        execution["google_project"] = _extract_antigravity_project_id(result.console_lines) or None
    elif _contains_antigravity_tool_schema_error(result.console_lines):
        execution["provider_error"] = "antigravity_tool_schema_rejected"
    if runtime_options is not None:
        source = _opencode_auth_source_from_runtime_options(runtime_options)
        execution["auth_source"] = source
        execution["auth_kind"] = str(getattr(runtime_options, "opencode_auth_kind", "") or "")
        execution["using_user_generation_key"] = bool(
            source != "opencode_builtin"
            and source != "account_auth"
            and getattr(runtime_options, "generation_api_key_configured", False)
        )
    return {
        "model": model,
        "mode": "opencode",
        "execution": execution,
    }


def _opencode_hybrid_ask_enabled() -> bool:
    return bool(getattr(get_settings(), "opencode_hybrid_ask_enabled", False))


def _opencode_hybrid_ask_usage_payload(
    result: OpenCodeExecutionResult,
    runtime_options: AgentRuntimeOptions | None = None,
) -> dict[str, Any]:
    model = str(result.model or get_settings().opencode_model or "opencode")
    return {
        "model": model,
        "mode": "opencode_hybrid_ask",
        "opencode": {
            "ok": result.ok,
            "run_id": result.run_id,
            "model": result.model,
            "exit_code": result.exit_code,
            "log_lines": len(result.console_lines),
            "auth_source": _opencode_auth_source_from_runtime_options(runtime_options),
            "auth_kind": str(getattr(runtime_options, "opencode_auth_kind", "") or ""),
        },
    }


def _stream_opencode_ask_generation(
    *,
    prompt: str,
    thread_id: str | None,
    trace_id: str,
    runtime_options: AgentRuntimeOptions | None,
    retrieval_state: dict[str, Any],
) -> Iterator[str]:
    executor = OpenCodeExecutor(
        provider_auth=_opencode_provider_auth_from_runtime_options(runtime_options),
        account_auth=_opencode_account_auth_from_runtime_options(runtime_options),
        mcp_servers=runtime_options.mcp_endpoints if runtime_options else None,
    )
    stream = executor.stream(
        prompt=prompt,
        thread_id=thread_id,
        trace_id=trace_id,
        task="ask",
        retrieved_context=str(retrieval_state.get("rag_result") or ""),
        citation_labels=list(retrieval_state.get("citation_labels") or []),
    )
    while True:
        try:
            event = next(stream)
        except StopIteration as stop:
            return stop.value
        event_type = str(event.get("type") or "")
        if event_type == "opencode_phase":
            label = str((event.get("content") or {}).get("label") or "").strip()
            if label:
                yield _sse({"type": "status", "message": label})
        elif event_type == "terminal_log":
            continue


def _artifact_execution_for_user(
    session: Session,
    *,
    user_id: str,
    thread_id: str,
    run_id: str,
) -> dict[str, Any]:
    execution = get_thread_execution(session, user_id=user_id, thread_id=thread_id, run_id=run_id)
    if not execution:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact run not found.")
    if artifact_expired(execution):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Artifact run has expired.")
    workspace = str(execution.get("workspace") or "").strip()
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact workspace not found.")
    return execution


def _artifact_access_error(exc: ArtifactAccessError) -> HTTPException:
    detail = str(exc) or "Artifact is not available."
    if "unsafe" in detail.lower() or "absolute" in detail.lower() or "escapes" in detail.lower():
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _require_internal_admin_token(x_internal_token: str | None) -> None:
    expected_token = get_settings().internal_api_token
    if not expected_token or x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def _cleanup_expired_artifacts() -> dict[str, Any]:
    settings = get_settings()
    removed = cleanup_expired_workspaces(
        settings.ontology_workdir / settings.opencode_workspace_subdir,
        retention_days=settings.opencode_artifact_retention_days,
    )
    if removed:
        logger.info(_log_event("assistant_artifact_cleanup", removed_workspaces=removed))
    return {
        "removed_workspaces": removed,
        "retention_days": settings.opencode_artifact_retention_days,
    }


def _artifact_filename(path: str, *, fallback: str = "artifact") -> str:
    try:
        clean_path = sanitize_artifact_path(path)
    except ArtifactAccessError:
        return fallback
    return clean_path.name or fallback


def _require_execution_path(execution: dict[str, Any], path: str) -> None:
    try:
        sanitize_artifact_path(path)
    except ArtifactAccessError as exc:
        raise _artifact_access_error(exc) from exc
    if not execution_allows_path(execution, path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file not found in this run.")


def _stream_agent_response(
    *,
    prompt: str,
    thread_id: str | None,
    agent_builder: Callable[[], OntoPortalAgent],
    requested_mode: str | None = None,
    on_completed: Callable[[dict[str, Any], str], None] | None = None,
    log_context: dict[str, Any] | None = None,
    opencode_resume: dict[str, Any] | None = None,
) -> Iterator[str]:
    stream_context = dict(log_context or {})
    stream_context.setdefault("thread_id", thread_id)
    stream_context.setdefault("prompt_chars", len(prompt))
    stream_context.setdefault("trace_id", uuid.uuid4().hex[:12])
    started_at = time.monotonic()
    logger.info(_log_event("assistant_stream_started", **stream_context))
    yield _sse({"type": "status", "message": "Assistant request received.", "thread_id": thread_id})
    final_state: dict[str, Any] = {}
    final_response_text = ""
    resolved_model = ""
    intent = "LEGACY"
    try:
        agent = agent_builder()
        runtime_options = getattr(agent, "runtime_options", None)

        if runtime_options is None:
            logger.info(_log_event("assistant_stream_mode", **stream_context, execution_mode="legacy_graph"))
            yield _sse({"type": "status", "message": "Executing ontology assistant pipeline..."})
            final_state = _collect_graph_final_state(agent=agent, prompt=prompt, thread_id=thread_id)
            for event in _emit_final_state(final_state):
                yield event
            final_response_text = str(final_state.get("final_response") or final_state.get("rag_result") or "").strip()
        else:
            llm = None if _uses_vertex_gemini_provider(runtime_options) else _build_chat_model(runtime_options)
            resolved_model = str(runtime_options.llm_model or get_settings().llm_model or "")
            model_candidates = _generation_model_candidates(
                resolved_model,
                generation_provider=getattr(runtime_options, "generation_provider", None),
                base_url=(runtime_options.openai_api_base if runtime_options else None) or get_settings().openai_api_base,
            )
            logger.info(_log_event("assistant_stream_mode", **stream_context, execution_mode="runtime", model=resolved_model))
            if _normalize_chat_mode(requested_mode) == "edit":
                intent = "EDIT"
                logger.info(
                    _log_event(
                        "assistant_stream_intent",
                        **stream_context,
                        intent=intent,
                        requested_mode="edit",
                        model=resolved_model,
                    )
                )
            else:
                yield _sse({"type": "status", "message": "Classifying request..."})
                intent = _classify_intent(llm, prompt)
                logger.info(_log_event("assistant_stream_intent", **stream_context, intent=intent, model=resolved_model))
            if intent == "EDIT":
                yield _sse({"type": "status", "message": "Starting OpenCode workspace..."})
                stream_kwargs: dict[str, Any] = {}
                if isinstance(opencode_resume, dict):
                    workspace = str(opencode_resume.get("workspace") or "").strip()
                    session_id = str(opencode_resume.get("session_id") or "").strip()
                    if workspace:
                        stream_kwargs["resume_workspace"] = workspace
                    if session_id:
                        stream_kwargs["resume_session_id"] = session_id
                opencode_result = yield from _stream_opencode_execution(
                    prompt=prompt,
                    thread_id=thread_id,
                    trace_id=stream_context["trace_id"],
                    runtime_options=runtime_options,
                    **stream_kwargs,
                )
                resolved_model = str(opencode_result.model or get_settings().opencode_model or resolved_model or "")
                final_state["generation_backend"] = "opencode"
                final_state["generation_usage"] = _opencode_usage_payload(opencode_result, runtime_options)
                final_state["citations"] = []
                yield _sse({"type": "usage", "content": final_state["generation_usage"]})
                provider_blocked = _contains_antigravity_provider_error(opencode_result.console_lines)
                if opencode_result.ok and not provider_blocked:
                    final_response_text = _opencode_success_response(opencode_result)
                    final_state["final_response"] = final_response_text
                    for chunk in _iter_text_chunks(final_response_text):
                        yield _sse({"type": "delta", "content": chunk})
                else:
                    final_response_text = _opencode_failure_response(opencode_result)
                    error_details = {
                        "trace_id": stream_context["trace_id"],
                        "status": "OpenCode workspace failed.",
                        "message": final_response_text,
                        "error_class": "OpenCodeProviderError" if provider_blocked else "OpenCodeExecutionError",
                        "status_code": 500,
                    }
                    final_state["generation_usage"]["error"] = error_details
                    final_state["final_response"] = final_response_text
                    logger.warning(
                        _log_event(
                            "assistant_stream_opencode_failed",
                            **stream_context,
                            intent=intent,
                            model=resolved_model or final_state["generation_usage"].get("model"),
                            workspace=opencode_result.workspace,
                            exit_code=opencode_result.exit_code,
                            changed_files=len(opencode_result.changed_files),
                        )
                    )
                    yield _sse({"type": "status", "message": "OpenCode workspace failed."})
                    yield _sse({"type": "error", "content": error_details})
                    for chunk in _iter_text_chunks(final_response_text):
                        yield _sse({"type": "delta", "content": chunk})
            else:
                yield _sse({"type": "status", "message": "Retrieving ontology context..."})
                final_state = _retrieve_runtime_state(prompt, runtime_options)
                logger.info(
                    _log_event(
                        "assistant_retrieval_complete",
                        **stream_context,
                        intent=intent,
                        model=resolved_model,
                        retrieval_backend=final_state.get("retrieval_backend"),
                        retrieval_chunk_count=final_state.get("retrieval_chunk_count"),
                        citations=len(final_state.get("citations") or []),
                        retrieval_error=bool(final_state.get("retrieval_error")),
                    )
                )
                if final_state.get("retrieval_error"):
                    yield _sse({"type": "status", "message": str(final_state["retrieval_error"])})
                if isinstance(final_state.get("citations"), list) and final_state.get("citations"):
                    yield _sse({"type": "citations", "content": final_state.get("citations")})

                if _opencode_hybrid_ask_enabled():
                    yield _sse({"type": "status", "message": "Generating answer with OpenCode..."})
                    try:
                        opencode_ask_result = yield from _stream_opencode_ask_generation(
                            prompt=prompt,
                            thread_id=thread_id,
                            trace_id=stream_context["trace_id"],
                            runtime_options=runtime_options,
                            retrieval_state=final_state,
                        )
                    except Exception as exc:
                        logger.warning(
                            _log_event(
                                "assistant_stream_hybrid_ask_failed",
                                **stream_context,
                                intent=intent,
                                model=get_settings().opencode_model,
                                **_failure_log_fields(exc),
                            )
                        )
                        yield _sse({"type": "status", "message": "OpenCode answer generation failed; using the standard model path."})
                    else:
                        if opencode_ask_result.ok and str(opencode_ask_result.final_text or "").strip():
                            resolved_model = str(opencode_ask_result.model or get_settings().opencode_model or resolved_model or "")
                            final_response_text = str(opencode_ask_result.final_text or "").strip()
                            final_state["final_response"] = final_response_text
                            final_state["generation_backend"] = "opencode:hybrid_ask"
                            final_state["generation_usage"] = _opencode_hybrid_ask_usage_payload(
                                opencode_ask_result,
                                runtime_options,
                            )
                            for chunk in _iter_text_chunks(final_response_text):
                                yield _sse({"type": "delta", "content": chunk})
                            yield _sse({"type": "usage", "content": final_state["generation_usage"]})
                            logger.info(
                                _log_event(
                                    "assistant_stream_hybrid_ask_completed",
                                    **stream_context,
                                    intent=intent,
                                    model=resolved_model,
                                    retrieval_backend=final_state.get("retrieval_backend"),
                                    citations=len(final_state.get("citations") or []),
                                )
                            )
                            model_candidates = []
                        else:
                            yield _sse({"type": "status", "message": "OpenCode answer generation produced no answer; using the standard model path."})

                if not final_state.get("final_response"):
                    yield _sse({"type": "status", "message": "Streaming answer..."})
                messages = _build_response_messages(
                    question=prompt,
                    rag_result=str(final_state.get("rag_result") or ""),
                    citations=list(final_state.get("citation_labels") or []),
                    retrieval_backend=str(final_state.get("retrieval_backend") or "unknown"),
                    retrieval_error=str(final_state.get("retrieval_error") or ""),
                )

                streamed_chunks: list[str] = []
                stream_usage: dict[str, Any] = {"model": resolved_model}
                streamed_reasoning: list[str] = []
                last_stream_error: Exception | None = None
                last_attempted_model = resolved_model

                for index, candidate_model in enumerate(model_candidates):
                    if index > 0:
                        logger.warning(
                            _log_event(
                                "assistant_stream_model_fallback",
                                **stream_context,
                                intent=intent,
                                previous_model=last_attempted_model,
                                fallback_model=candidate_model,
                            )
                        )
                        yield _sse(
                            {
                                "type": "status",
                                "message": f"Primary model unavailable. Switching to {candidate_model}.",
                            }
                        )

                    candidate_attempts = _stream_attempts_for_model(
                        candidate_model,
                        base_url=runtime_options.openai_api_base,
                    )
                    candidate_succeeded = False

                    for attempt_index in range(candidate_attempts):
                        candidate_llm = _build_chat_model(runtime_options, model_override=candidate_model)
                        candidate_usage: dict[str, Any] = {"model": candidate_model}
                        candidate_chunks: list[str] = []
                        candidate_reasoning: list[str] = []

                        if attempt_index > 0:
                            logger.warning(
                                _log_event(
                                    "assistant_stream_model_retry",
                                    **stream_context,
                                    intent=intent,
                                    model=candidate_model,
                                    attempt=attempt_index + 1,
                                    attempts_total=candidate_attempts,
                                )
                            )
                            yield _sse(
                                {
                                    "type": "status",
                                    "message": f"Retrying {candidate_model} after a transient provider failure.",
                                }
                            )
                            time.sleep(_stream_retry_delay_seconds(attempt_index))

                        try:
                            if _uses_vertex_gemini_provider(runtime_options):
                                for event in _stream_vertex_gemini_events(
                                    runtime_options=runtime_options,
                                    model=candidate_model,
                                    messages=messages,
                                    usage_state=candidate_usage,
                                    answer_chunks=candidate_chunks,
                                    reasoning_chunks=candidate_reasoning,
                                ):
                                    yield _sse(event)
                            elif _uses_google_openai_base(runtime_options.openai_api_base):
                                for event in _stream_openai_compatible_events(
                                    runtime_options=runtime_options,
                                    model=candidate_model,
                                    messages=messages,
                                    usage_state=candidate_usage,
                                    answer_chunks=candidate_chunks,
                                    reasoning_chunks=candidate_reasoning,
                                ):
                                    yield _sse(event)
                            else:
                                try:
                                    for chunk in candidate_llm.stream(messages):
                                        text = _flatten_chunk_text(getattr(chunk, "content", chunk))
                                        if not text:
                                            continue
                                        candidate_chunks.append(text)
                                        yield _sse({"type": "delta", "content": text})
                                except Exception as exc:  # pragma: no cover - fallback path
                                    logger.warning(
                                        _log_event(
                                            "assistant_stream_chunk_fallback",
                                            **stream_context,
                                            intent=intent,
                                            model=candidate_model,
                                            **_failure_log_fields(exc),
                                        )
                                    )
                                    reply = candidate_llm.invoke(messages)
                                    candidate_usage.update(_extract_generation_usage(reply))
                                    text = _flatten_chunk_text(getattr(reply, "content", reply)).strip()
                                    candidate_chunks = []
                                    for chunk in _iter_text_chunks(text):
                                        candidate_chunks.append(chunk)
                                        yield _sse({"type": "delta", "content": chunk})

                            if not candidate_chunks:
                                reply = candidate_llm.invoke(messages)
                                candidate_usage.update(_extract_generation_usage(reply))
                                text = _flatten_chunk_text(getattr(reply, "content", reply)).strip()
                                candidate_chunks = []
                                for chunk in _iter_text_chunks(text):
                                    candidate_chunks.append(chunk)
                                    yield _sse({"type": "delta", "content": chunk})

                            streamed_chunks = candidate_chunks
                            streamed_reasoning = candidate_reasoning
                            stream_usage = candidate_usage
                            resolved_model = candidate_model
                            last_attempted_model = candidate_model
                            candidate_succeeded = True
                            break
                        except Exception as exc:
                            last_stream_error = exc
                            last_attempted_model = candidate_model
                            logger.warning(
                                _log_event(
                                    "assistant_stream_model_attempt_failed",
                                    **stream_context,
                                    intent=intent,
                                    model=candidate_model,
                                    attempt=attempt_index + 1,
                                    attempts_total=candidate_attempts,
                                    response_started=bool(candidate_chunks),
                                    reasoning_started=bool(candidate_reasoning),
                                    error=str(exc),
                                    **_failure_log_fields(exc),
                                )
                            )
                            if candidate_reasoning and not candidate_chunks:
                                yield _sse({"type": "reasoning_reset"})
                            if candidate_chunks:
                                raise
                            if attempt_index < candidate_attempts - 1 and _can_retry_with_fallback_model(exc):
                                continue
                            if index == len(model_candidates) - 1 or not _can_retry_with_fallback_model(exc):
                                raise
                            break

                    if candidate_succeeded:
                        break

                if not streamed_chunks and last_stream_error is not None:
                    raise last_stream_error

                if final_state.get("final_response"):
                    final_response_text = str(final_state.get("final_response") or "").strip()
                else:
                    final_response_text = "".join(streamed_chunks).strip() or "No response generated."
                    final_state["final_response"] = final_response_text
                    final_state["generation_backend"] = f"llm:{resolved_model}"
                    final_state["generation_usage"] = stream_usage
                    if streamed_reasoning:
                        final_state["generation_reasoning"] = "".join(streamed_reasoning).strip()
                    yield _sse({"type": "usage", "content": stream_usage})
        logger.info(
            _log_event(
                "assistant_stream_completed",
                **stream_context,
                intent=intent,
                model=resolved_model or final_state.get("generation_backend"),
                duration_ms=_elapsed_ms(started_at),
                response_chars=len(final_response_text),
                citations=len(final_state.get("citations") or []),
                retrieval_backend=final_state.get("retrieval_backend"),
                retrieval_error=bool(final_state.get("retrieval_error")),
            )
        )
    except Exception as exc:  # pragma: no cover - smoke tests cover happy path.
        status_message, client_message = _stream_failure_payload(exc)
        error_details = {
            "trace_id": stream_context["trace_id"],
            "status": status_message,
            "message": client_message,
            **_failure_log_fields(exc),
        }
        failure_context = {
            **stream_context,
            "intent": intent,
            "model": resolved_model or get_settings().llm_model,
            "duration_ms": _elapsed_ms(started_at),
            **_failure_log_fields(exc),
        }
        if error_details["rate_limited"]:
            logger.warning(_log_event("assistant_stream_failed", **failure_context))
        else:
            logger.exception(_log_event("assistant_stream_failed", **failure_context))

        if on_completed is not None:
            try:
                on_completed(
                    {
                        "generation_usage": {
                            "model": resolved_model or get_settings().llm_model,
                            "error": error_details,
                        },
                        "generation_reasoning": "",
                        "citations": [],
                        "retrieval_backend": final_state.get("retrieval_backend") if final_state else "",
                        "retrieval_error": final_state.get("retrieval_error") if final_state else "",
                    },
                    client_message,
                )
            except Exception as persist_exc:  # pragma: no cover - persistence failure is secondary.
                logger.exception(
                    _log_event(
                        "assistant_stream_failure_persist_failed",
                        **stream_context,
                        duration_ms=_elapsed_ms(started_at),
                        error_class=persist_exc.__class__.__name__,
                    )
                )

        yield _sse({"type": "status", "message": status_message})
        yield _sse({"type": "error", "content": error_details})
        yield _sse({"type": "delta", "content": client_message})
        yield _sse_done()
        return

    if on_completed is not None:
        try:
            on_completed(final_state, final_response_text or "No response generated.")
        except Exception as exc:  # pragma: no cover - persistence failure should not poison the response.
            logger.exception(
                _log_event(
                    "assistant_stream_persist_failed",
                    **stream_context,
                    duration_ms=_elapsed_ms(started_at),
                    error_class=exc.__class__.__name__,
                )
            )
    yield _sse_done()


@app.on_event("startup")
def startup() -> None:
    init_db()
    _cleanup_expired_artifacts()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/admin/artifacts/cleanup")
def admin_artifact_cleanup(x_internal_token: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _require_internal_admin_token(x_internal_token)
    return _cleanup_expired_artifacts()


@app.get("/api/v1/me/artifacts/{thread_id}/{run_id}/files")
def me_artifact_files(
    thread_id: str,
    run_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    execution = _artifact_execution_for_user(
        session,
        user_id=user_context.user_id,
        thread_id=thread_id,
        run_id=run_id,
    )
    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "expires_at": execution.get("expires_at"),
        "workspace": execution.get("workspace"),
        "files": list_artifact_files(execution),
        "bundle_url": f"/assistant/artifacts/{thread_id}/{run_id}/bundle.zip",
    }


@app.get("/api/v1/me/artifacts/{thread_id}/{run_id}/file")
def me_artifact_file(
    thread_id: str,
    run_id: str,
    request: Request,
    path: str = Query(..., min_length=1),
    view: str = Query(default="file"),
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    execution = _artifact_execution_for_user(
        session,
        user_id=user_context.user_id,
        thread_id=thread_id,
        run_id=run_id,
    )
    _require_execution_path(execution, path)
    try:
        if str(view or "").lower() == "diff":
            payload = read_artifact_diff(
                execution["workspace"],
                path,
                max_chars=max(10_000, int(get_settings().opencode_max_diff_chars)),
            )
            payload["view"] = "diff"
            return payload

        payload = read_artifact_text(execution["workspace"], path)
        payload["view"] = "file"
        return payload
    except ArtifactAccessError as exc:
        raise _artifact_access_error(exc) from exc


@app.get("/api/v1/me/artifacts/{thread_id}/{run_id}/download")
def me_artifact_download(
    thread_id: str,
    run_id: str,
    request: Request,
    path: str = Query(..., min_length=1),
    session: Session = Depends(get_db_session),
) -> FileResponse:
    user_context = _resolve_user_context(request)
    execution = _artifact_execution_for_user(
        session,
        user_id=user_context.user_id,
        thread_id=thread_id,
        run_id=run_id,
    )
    _require_execution_path(execution, path)
    try:
        file_path = resolve_artifact_file(execution["workspace"], path)
    except ArtifactAccessError as exc:
        raise _artifact_access_error(exc) from exc
    return FileResponse(
        path=str(file_path),
        filename=_artifact_filename(path),
        media_type=file_metadata(execution["workspace"], path).get("content_type") or "application/octet-stream",
    )


@app.get("/api/v1/me/artifacts/{thread_id}/{run_id}/bundle.zip")
def me_artifact_bundle(
    thread_id: str,
    run_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
) -> Response:
    user_context = _resolve_user_context(request)
    execution = _artifact_execution_for_user(
        session,
        user_id=user_context.user_id,
        thread_id=thread_id,
        run_id=run_id,
    )
    try:
        payload = build_artifact_bundle(execution["workspace"], execution)
    except ArtifactAccessError as exc:
        raise _artifact_access_error(exc) from exc
    filename = f"matportal-assistant-{run_id}.zip"
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/v1/me/bootstrap")
def me_bootstrap(
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    _cleanup_old_history(session, user_id=user_context.user_id)
    settings_payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=False)
    return {
        "user": {
            "id": user_context.user_id,
            "username": user_context.username,
            "email": user_context.email,
        },
        "features": {
            "settings": True,
            "history": True,
            "mcp": True,
        },
        "settings": _serialize_settings_for_output(settings_payload),
        "threads": [_serialize_thread(item) for item in list_threads(session, user_id=user_context.user_id)],
        "thread_count": len(list_threads(session, user_id=user_context.user_id, limit=500)),
    }


@app.get("/api/v1/me/settings")
def me_get_settings(
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    return _serialize_settings_for_output(payload)


@app.get("/api/v1/me/skills")
def me_get_skills(
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    skills = _assistant_installed_skills(payload)
    return {
        "skills": skills,
        "installed_count": len(skills),
        "enabled_count": len([item for item in skills if item.get("enabled")]),
    }


@app.put("/api/v1/me/settings")
def me_put_settings(
    payload: AssistantSettingsIn,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    encryption_service = _encryption_required()

    existing_payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    raw_payload = payload.model_dump()
    normalized_payload = _normalize_settings_payload(raw_payload)
    raw_opencode = raw_payload.get("opencode", {})
    incoming_opencode = normalized_payload.get("opencode", {})
    existing_opencode = existing_payload.get("opencode", {})
    if bool(raw_opencode.get("clear_account_auth", False)):
        incoming_opencode["auth_json"] = ""
        incoming_opencode["codex_auth_json"] = ""
    else:
        if str(incoming_opencode.get("auth_json", "")).strip() == "__configured__":
            incoming_opencode["auth_json"] = ""
        if str(incoming_opencode.get("codex_auth_json", "")).strip() == "__configured__":
            incoming_opencode["codex_auth_json"] = ""
        if not str(incoming_opencode.get("auth_json", "")).strip() and _has_persisted_secret(existing_opencode.get("auth_json", "")):
            incoming_opencode["auth_json"] = str(existing_opencode.get("auth_json", "")).strip()
        if not str(incoming_opencode.get("codex_auth_json", "")).strip() and _has_persisted_secret(existing_opencode.get("codex_auth_json", "")):
            incoming_opencode["codex_auth_json"] = str(existing_opencode.get("codex_auth_json", "")).strip()
    incoming_opencode["auth_json"] = _validate_auth_json(incoming_opencode.get("auth_json", ""), label="OpenCode auth JSON")
    incoming_opencode["codex_auth_json"] = _validate_auth_json(incoming_opencode.get("codex_auth_json", ""), label="Codex auth JSON")
    normalized_payload["opencode"] = incoming_opencode
    for provider_key in ("generation", "embeddings", "reranker"):
        incoming_provider = normalized_payload.get(provider_key, {})
        existing_provider = existing_payload.get(provider_key, {})
        raw_provider = raw_payload.get(provider_key, {})
        if bool(raw_provider.get("clear_api_key", False)):
            incoming_provider["api_key"] = ""
            continue
        incoming_provider_name = str(incoming_provider.get("provider", "")).strip().lower()
        existing_provider_name = str(existing_provider.get("provider", "")).strip().lower()
        if (
            incoming_provider_name == existing_provider_name
            and not str(incoming_provider.get("api_key", "")).strip()
            and _has_persisted_secret(existing_provider.get("api_key", ""))
        ):
            incoming_provider["api_key"] = str(existing_provider.get("api_key", "")).strip()

    existing_mcp_by_identity = {
        (item.get("name", "").strip(), item.get("url", "").strip()): item
        for item in existing_payload.get("mcp_servers", [])
    }
    for server in normalized_payload.get("mcp_servers", []):
        identity = (server.get("name", "").strip(), server.get("url", "").strip())
        existing_server = existing_mcp_by_identity.get(identity)
        auth_mode = _normalize_mcp_auth_mode(
            server.get("auth_mode"),
            api_key=str(server.get("api_key", "") or ""),
            username=str(server.get("username", "") or ""),
            password=str(server.get("password", "") or ""),
        )
        server["auth_mode"] = auth_mode
        if auth_mode == _MCP_AUTH_API_KEY:
            server["username"] = ""
            server["password"] = ""
            if (
                existing_server
                and _normalize_mcp_auth_mode(existing_server.get("auth_mode"), api_key=existing_server.get("api_key")) == _MCP_AUTH_API_KEY
                and not str(server.get("api_key", "")).strip()
                and _has_persisted_secret(existing_server.get("api_key", ""))
            ):
                server["api_key"] = str(existing_server.get("api_key", "")).strip()
            continue
        if auth_mode == _MCP_AUTH_BASIC_USER:
            server["api_key"] = ""
            if (
                existing_server
                and _normalize_mcp_auth_mode(
                    existing_server.get("auth_mode"),
                    api_key=existing_server.get("api_key"),
                    username=existing_server.get("username"),
                    password=existing_server.get("password"),
                )
                == _MCP_AUTH_BASIC_USER
                and not str(server.get("password", "")).strip()
                and _has_persisted_secret(existing_server.get("password", ""))
            ):
                server["password"] = str(existing_server.get("password", "")).strip()
            continue
        server["api_key"] = ""
        server["username"] = ""
        server["password"] = ""

    settings_blob = {
        "generation": normalized_payload["generation"],
        "embeddings": normalized_payload["embeddings"],
        "reranker": normalized_payload["reranker"],
        "retrieval": normalized_payload["retrieval"],
        "opencode": normalized_payload["opencode"],
    }
    encrypted_payload, key_version = encryption_service.encrypt_json(settings_blob)
    upsert_user_settings(
        session,
        user_id=user_context.user_id,
        settings_encrypted=encrypted_payload,
        key_version=key_version,
    )

    replace_mcp_servers(
        session,
        user_id=user_context.user_id,
        mcp_servers=normalized_payload["mcp_servers"],
        encrypt_secret=lambda secret_payload: encryption_service.encrypt_json(secret_payload),
    )
    return _serialize_settings_for_output(normalized_payload)


@app.post("/api/v1/me/auth/codex/start")
def me_codex_auth_start(request: Request) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    try:
        auth_session = _account_auth_manager.start_codex_device_auth(user_id=user_context.user_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Codex CLI is not installed in the assistant runtime.",
        ) from exc
    return {
        "auth_session_id": auth_session.id,
        "provider": "codex",
        "login_url": auth_session.login_url,
        "user_code": auth_session.user_code,
        "expires_at": auth_session.expires_at.isoformat(),
        "status": "pending",
    }


@app.get("/api/v1/me/auth/codex/status")
def me_codex_auth_status(
    auth_session_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    try:
        auth_session = _account_auth_manager.get(
            session_id=auth_session_id,
            user_id=user_context.user_id,
            provider="codex",
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Codex login session was not found.") from exc
    auth_path = _account_auth_manager.codex_auth_json_path(auth_session)
    if auth_path is None:
        failed = bool(auth_session.process and auth_session.process.poll() is not None)
        return {
            "auth_session_id": auth_session.id,
            "provider": "codex",
            "status": "failed" if failed else "pending",
            "login_url": auth_session.login_url,
            "user_code": auth_session.user_code,
            "expires_at": auth_session.expires_at.isoformat(),
            "message": "Codex login is still waiting for browser confirmation." if not failed else "Codex login exited before saving auth.",
        }

    try:
        codex_auth_json = load_json_object(auth_path)
        settings_payload = _persist_opencode_account_auth(
            session,
            user_id=user_context.user_id,
            auth_kind="codex",
            codex_auth_json=codex_auth_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    finally:
        _account_auth_manager.pop(session_id=auth_session.id, user_id=user_context.user_id, provider="codex")
        _account_auth_manager.finish(auth_session)
    return {
        "auth_session_id": auth_session.id,
        "provider": "codex",
        "status": "connected",
        "settings": settings_payload,
    }


@app.post("/api/v1/me/auth/antigravity/start")
def me_antigravity_auth_start(payload: AntigravityAuthStartIn, request: Request) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    try:
        auth_session = _account_auth_manager.start_antigravity(
            user_id=user_context.user_id,
            project_id=payload.project_id or "",
        )
    except AntigravityConfigError as exc:
        logger.warning(
            "Gemini Antigravity auth start blocked by missing OAuth config user=%s oauth=%s",
            user_context.user_id,
            antigravity_oauth_config_summary(),
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    logger.info(
        "Gemini Antigravity auth start user=%s session=%s project_set=%s oauth=%s",
        user_context.user_id,
        auth_session.id,
        bool(auth_session.project_id),
        antigravity_oauth_config_summary(),
    )
    return {
        "auth_session_id": auth_session.id,
        "provider": "gemini_antigravity",
        "login_url": auth_session.login_url,
        "expires_at": auth_session.expires_at.isoformat(),
        "callback_required": True,
        "redirect_uri": antigravity_redirect_uri(),
        "status": "pending",
    }


@app.post("/api/v1/me/auth/antigravity/complete")
def me_antigravity_auth_complete(
    payload: AntigravityAuthCompleteIn,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    completed = False
    try:
        auth_session = _account_auth_manager.get(
            session_id=payload.auth_session_id,
            user_id=user_context.user_id,
            provider="gemini_antigravity",
        )
        callback_text = str(payload.callback_url_or_code or "").strip()
        parsed_callback = urlparse(callback_text)
        callback_query = parse_qs(parsed_callback.query) if parsed_callback.scheme and parsed_callback.netloc else {}
        logger.info(
            "Gemini Antigravity auth complete received user=%s session=%s callback_url=%s callback_has_code=%s callback_has_state=%s callback_query_keys=%s",
            user_context.user_id,
            payload.auth_session_id,
            bool(parsed_callback.scheme and parsed_callback.netloc),
            bool((callback_query.get("code") or [""])[0]) if callback_query else bool(callback_text),
            bool((callback_query.get("state") or [""])[0]) if callback_query else False,
            sorted(key for key in callback_query.keys() if key not in {"code"}),
        )
        code, _state = parse_antigravity_callback(
            payload.callback_url_or_code,
            expected_state=auth_session.state,
        )
        opencode_auth = exchange_antigravity_code(
            code=code,
            code_verifier=auth_session.code_verifier,
            project_id=auth_session.project_id,
        )
        settings_payload = _persist_opencode_account_auth(
            session,
            user_id=user_context.user_id,
            auth_kind="gemini_antigravity",
            opencode_auth_json=json.dumps(opencode_auth, separators=(",", ":"), ensure_ascii=False),
        )
        completed = True
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Gemini Antigravity login session was not found.") from exc
    except AntigravityConfigError as exc:
        logger.warning(
            "Gemini Antigravity auth completion blocked by missing OAuth config user=%s oauth=%s",
            user_context.user_id,
            antigravity_oauth_config_summary(),
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        logger.info(
            "Gemini Antigravity auth completion failed for user=%s reason=%s",
            user_context.user_id,
            _sanitize_provider_error(str(exc)),
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    finally:
        if completed:
            try:
                auth_session = _account_auth_manager.pop(
                    session_id=payload.auth_session_id,
                    user_id=user_context.user_id,
                    provider="gemini_antigravity",
                )
                _account_auth_manager.finish(auth_session)
            except KeyError:
                pass
    return {
        "auth_session_id": payload.auth_session_id,
        "provider": "gemini_antigravity",
        "status": "connected",
        "settings": settings_payload,
    }


@app.get("/api/v1/me/auth/antigravity/models")
def me_antigravity_auth_models(
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    settings_payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    opencode = _normalize_opencode_settings(settings_payload.get("opencode", {}))
    connected = (
        opencode.get("auth_source") == "account_auth"
        and opencode.get("auth_kind") == "gemini_antigravity"
        and _has_persisted_secret(opencode.get("auth_json"))
    )
    selected = normalize_antigravity_model_ref(
        opencode.get("antigravity_model"),
        default=str(get_settings().opencode_antigravity_model or "") or DEFAULT_ANTIGRAVITY_MODEL_REF,
    )
    return {
        "provider": "gemini_antigravity",
        "connected": connected,
        "default_model": normalize_antigravity_model_ref(get_settings().opencode_antigravity_model),
        "selected_model": selected,
        "models": antigravity_model_options(selected_model_ref=selected),
    }


@app.post("/api/v1/me/settings/provider/check")
def me_check_settings_provider(
    payload: ProviderCheckIn,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    existing_payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    defaults = _default_settings_payload()
    scope = str(payload.scope or "generation").strip().lower()
    if scope not in ("generation", "embeddings", "reranker"):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unsupported provider scope.")

    raw_provider = payload.model_dump()
    provider = _normalize_provider(raw_provider, defaults.get(scope, defaults["generation"]))
    existing_provider = existing_payload.get(scope, {})
    incoming_name = str(provider.get("provider", "")).strip().lower()
    existing_name = str(existing_provider.get("provider", "")).strip().lower()
    if (
        not str(provider.get("api_key", "")).strip()
        and incoming_name == existing_name
        and _has_persisted_secret(existing_provider.get("api_key", ""))
    ):
        provider["api_key"] = str(existing_provider.get("api_key", "")).strip()
    if _uses_gemini_api_provider(provider.get("provider")) and not str(provider.get("base_url") or "").strip():
        provider["base_url"] = _GOOGLE_GEMINI_OPENAI_BASE_URL

    clean_provider = str(provider.get("provider") or "").strip().lower()
    if clean_provider == "vertex_gemini":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Vertex service-account checks are not exposed in per-user settings.",
        )
    if clean_provider in ("none", ""):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No provider is configured.")

    result = _check_openai_compatible_provider(
        api_key=str(provider.get("api_key") or ""),
        base_url=str(provider.get("base_url") or ""),
        model=str(provider.get("model") or ""),
    )
    result.update(
        {
            "provider": provider.get("provider"),
            "model": provider.get("model") or "",
            "base_url": provider.get("base_url") or "",
        }
    )
    return result


@app.get("/api/v1/me/threads")
def me_list_threads(
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    threads = [_serialize_thread(item) for item in list_threads(session, user_id=user_context.user_id)]
    return {"threads": threads}


@app.post("/api/v1/me/threads")
def me_create_thread(
    payload: ThreadCreateRequest,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    thread = create_thread(
        session,
        user_id=user_context.user_id,
        title=payload.title,
        thread_id=payload.thread_id or str(uuid.uuid4()),
    )
    return _serialize_thread(thread)


@app.get("/api/v1/me/threads/{thread_id}/messages")
def me_get_thread_messages(
    thread_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    messages = [
        _serialize_message(item)
        for item in list_thread_messages(session, user_id=user_context.user_id, thread_id=thread_id)
    ]
    return {"thread_id": thread_id, "messages": messages}


@app.delete("/api/v1/me/threads/{thread_id}")
def me_delete_thread(
    thread_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    deleted = delete_thread(session, user_id=user_context.user_id, thread_id=thread_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")
    return {"deleted": True, "thread_id": thread_id}


@app.get("/api/v1/me/mcp/health")
def me_mcp_health(
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    settings = get_settings()
    user_context = _resolve_user_context(request)
    payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    servers = payload.get("mcp_servers", [])
    health = []
    for item in servers:
        endpoint = str(item.get("url", "")).strip()
        if not endpoint:
            continue
        enabled = bool(item.get("enabled", True))
        timeout_ms = _normalized_mcp_timeout_ms(item.get("timeout_ms"), url=endpoint)
        if not enabled:
            health.append(
                {
                    "name": item.get("name") or endpoint,
                    "url": endpoint,
                    "enabled": False,
                    "ok": False,
                    "error": "disabled",
                }
            )
            continue
        try:
            headers = _runtime_mcp_headers_for_server(item, settings)
            client = McpClient(
                [
                    {
                        "url": endpoint,
                        "headers": headers or None,
                        "timeout_ms": timeout_ms,
                    }
                ]
            )
            tools = client.list_tools()
            health.append(
                {
                    "name": item.get("name") or endpoint,
                    "url": endpoint,
                    "enabled": True,
                    "ok": True,
                    "tools": len(tools),
                }
            )
        except McpInvocationError as exc:
            health.append(
                {
                    "name": item.get("name") or endpoint,
                    "url": endpoint,
                    "enabled": True,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return {"servers": health}


@app.post("/api/v1/me/chat/stream")
def me_chat_stream(
    payload: ChatStreamRequest,
    request: Request,
    x_internal_token: Optional[str] = Header(default=None),
    session: Session = Depends(get_db_session),
) -> StreamingResponse:
    settings = get_settings()
    expected_token = settings.internal_api_token
    if expected_token and x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Prompt cannot be blank")

    user_context = _resolve_user_context(request)
    _cleanup_old_history(session, user_id=user_context.user_id)
    thread = ensure_thread(
        session,
        user_id=user_context.user_id,
        thread_id=payload.thread_id,
        title=payload.thread_title,
    )
    create_message(
        session,
        user_id=user_context.user_id,
        thread_id=thread.thread_id,
        role="user",
        content=prompt,
    )
    settings_payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    runtime_options = _runtime_options_from_settings(settings_payload)
    requested_title = payload.thread_title or _derive_thread_title(prompt)
    if not (thread.title or "").strip():
        update_thread_title(
            session,
            user_id=user_context.user_id,
            thread_id=thread.thread_id,
            title=requested_title,
        )
    log_context = {
        "trace_id": uuid.uuid4().hex[:12],
        "user_id": _compact_user_id(user_context.user_id),
        "username": user_context.username,
        "thread_id": thread.thread_id,
        "follow_up": bool(payload.thread_id),
        "request_id": request.headers.get("x-request-id") or request.headers.get("X-Request-Id"),
    }
    latest_execution = get_latest_thread_execution(
        session,
        user_id=user_context.user_id,
        thread_id=thread.thread_id,
    )
    opencode_resume: dict[str, Any] | None = None
    if isinstance(latest_execution, dict) and not artifact_expired(latest_execution):
        resume_workspace = str(latest_execution.get("workspace") or "").strip()
        resume_session_id = str(latest_execution.get("session_id") or "").strip()
        if resume_workspace or resume_session_id:
            opencode_resume = {
                "workspace": resume_workspace,
                "session_id": resume_session_id,
            }

    def on_completed(final_state: dict[str, Any], final_response_text: str) -> None:
        create_message(
            session,
            user_id=user_context.user_id,
            thread_id=thread.thread_id,
            role="assistant",
            content=final_response_text or "No response generated.",
            reasoning_summary=_persistable_reasoning_summary(final_state),
            usage_json=final_state.get("generation_usage") if isinstance(final_state.get("generation_usage"), dict) else {},
            citations_json=final_state.get("citations") if isinstance(final_state.get("citations"), list) else [],
        )

    return StreamingResponse(
        _stream_agent_response(
            prompt=prompt,
            thread_id=thread.thread_id,
            agent_builder=lambda: OntoPortalAgent(runtime_options=runtime_options),
            requested_mode=payload.mode,
            on_completed=on_completed,
            log_context=log_context,
            opencode_resume=opencode_resume,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/chat/stream")
def chat_stream(
    request: ChatStreamRequest,
    x_internal_token: Optional[str] = Header(default=None),
) -> StreamingResponse:
    settings = get_settings()
    expected_token = settings.internal_api_token
    if expected_token and x_internal_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )

    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be blank",
        )

    return StreamingResponse(
        _stream_agent_response(
            prompt=prompt,
            thread_id=request.thread_id,
            agent_builder=lambda: _get_agent(),
            requested_mode=request.mode,
            log_context={
                "trace_id": uuid.uuid4().hex[:12],
                "thread_id": request.thread_id,
                "follow_up": bool(request.thread_id),
                "legacy": True,
            },
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    uvicorn.run("ontoportal_agent.server:app", host="0.0.0.0", port=8090, reload=False)


if __name__ == "__main__":
    main()
