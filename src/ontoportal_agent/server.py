from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable, Iterator, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .agent.graph import _extract_generation_reasoning, _extract_generation_usage
from .agent.options import AgentRuntimeOptions
from .agent.runtime import OntoPortalAgent
from .config import get_settings
from .db import EncryptionService, init_db
from .db.base import get_db_session
from .db.models import AssistantMessage
from .db.repositories import (
    create_message,
    create_thread,
    delete_thread,
    ensure_thread,
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
from .rag_client import RagClient

logger = logging.getLogger("uvicorn.error").getChild("ontoportal_agent")

app = FastAPI(
    title="OntoPortal Agent API",
    description="Streaming bridge for the MatPortal assistant UI.",
    version="2.0.0",
)

_agent_lock = Lock()
_agent_instance: Optional[OntoPortalAgent] = None

_LEGACY_DEFAULT_MCP_TIMEOUT_MS = 10_000
_BUILTIN_DEFAULT_MCP_TIMEOUT_MS = 30_000


class ChatStreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    thread_id: Optional[str] = None
    thread_title: Optional[str] = None


class ProviderConfigIn(BaseModel):
    provider: str = "openai_compatible"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class McpServerIn(BaseModel):
    name: str
    url: str
    api_key: Optional[str] = None
    enabled: bool = True
    timeout_ms: int = _BUILTIN_DEFAULT_MCP_TIMEOUT_MS


class RetrievalSettingsIn(BaseModel):
    chunk_count: int = Field(default=20, ge=1, le=40)


class AssistantSettingsIn(BaseModel):
    generation: ProviderConfigIn = Field(default_factory=ProviderConfigIn)
    embeddings: ProviderConfigIn = Field(default_factory=ProviderConfigIn)
    reranker: ProviderConfigIn = Field(default_factory=lambda: ProviderConfigIn(provider="none"))
    retrieval: RetrievalSettingsIn = Field(default_factory=RetrievalSettingsIn)
    mcp_servers: list[McpServerIn] = Field(default_factory=list)


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


def _failure_log_fields(exc: Exception) -> dict[str, Any]:
    return {
        "error_class": exc.__class__.__name__,
        "status_code": _error_status_code(exc),
        "retry_after_seconds": _error_retry_after_seconds(exc),
        "rate_limited": _is_rate_limit_error(exc),
    }


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


def _build_chat_model(runtime_options: AgentRuntimeOptions | None) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=(runtime_options.openai_api_key if runtime_options else "") or settings.openai_api_key,
        base_url=(runtime_options.openai_api_base if runtime_options else None) or settings.openai_api_base,
        model=(runtime_options.llm_model if runtime_options else None) or settings.llm_model,
        temperature=0.0,
    )


def _classify_intent(llm: ChatOpenAI, prompt: str) -> str:
    return classify_user_intent(prompt, llm=llm)


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


def _build_reasoning_messages(*, question: str, answer: str, citations: list[str]) -> list[Any]:
    citation_text = "\n".join(f"- {item}" for item in citations) if citations else "- none"
    return [
        SystemMessage(
            content=(
                "Write a concise reasoning summary for the user-visible answer.\n"
                "Rules:\n"
                "- Use 2-4 short bullet points.\n"
                "- Explain why the answer was selected.\n"
                "- Do not reveal hidden chain-of-thought.\n"
                "- Do not introduce facts outside the provided answer/citations."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\n"
                f"Answer: {answer}\n"
                f"Citations:\n{citation_text}"
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
        "rag_result": "",
        "retrieval_backend": "none",
        "retrieval_error": "",
        "retrieval_chunk_count": rag_top_k,
    }

    if rag_base_url and rag_query_path:
        try:
            result = RagClient(base_url=rag_base_url, query_path=rag_query_path).query(prompt, top_k=rag_top_k)
            state["rag_result"] = result.answer
            state["citations"] = [f"{src.ontology_id} v{src.version}" for src in result.sources]
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
            state["citations"] = [
                f"{src.get('ontology_id', 'unknown')} v{src.get('version', 'unknown')}"
                for src in sources
            ]
            state["retrieval_backend"] = "mcp"
            return state
        except (McpInvocationError, KeyError, TypeError, ValueError) as err:
            state["retrieval_backend"] = "none"
            existing_error = state.get("retrieval_error")
            state["retrieval_error"] = f"{existing_error}; fallback failed: {err}" if existing_error else str(err)

    state["retrieval_backend"] = "none"
    state["rag_result"] = ""
    state["citations"] = []

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
    model_reasoning = final_state.get("generation_reasoning")
    if model_reasoning:
        yield _sse({"type": "model_reasoning", "content": str(model_reasoning)})

    generation_usage = final_state.get("generation_usage")
    if isinstance(generation_usage, dict):
        yield _sse({"type": "usage", "content": generation_usage})

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
    return {
        "generation": {
            "provider": settings.default_generation_provider,
            "model": settings.default_generation_model or settings.llm_model,
            "api_key": "",
            "base_url": settings.default_generation_base_url or settings.openai_api_base or "",
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
        "mcp_servers": [
            {
                "name": f"MCP {index + 1}",
                "url": endpoint,
                "api_key": "",
                "enabled": True,
                "timeout_ms": _BUILTIN_DEFAULT_MCP_TIMEOUT_MS,
            }
            for index, endpoint in enumerate(mcp_endpoints)
        ],
    }


def _normalize_provider(provider_payload: dict[str, Any], default_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(default_payload)
    normalized.update(
        {
            "provider": str(provider_payload.get("provider", default_payload.get("provider", "openai_compatible"))),
            "model": str(provider_payload.get("model", default_payload.get("model", "")) or ""),
            "api_key": str(provider_payload.get("api_key", "") or ""),
            "base_url": str(provider_payload.get("base_url", default_payload.get("base_url", "")) or ""),
        }
    )
    return normalized


def _normalize_settings_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    defaults = _default_settings_payload()
    generation = _normalize_provider(raw_payload.get("generation", {}), defaults["generation"])
    embeddings = _normalize_provider(raw_payload.get("embeddings", {}), defaults["embeddings"])
    reranker = _normalize_provider(raw_payload.get("reranker", {}), defaults["reranker"])
    retrieval = {
        "chunk_count": _normalized_chunk_count(
            raw_payload.get("retrieval", {}).get("chunk_count"),
            default=int(defaults["retrieval"]["chunk_count"]),
        )
    }

    mcp_servers = []
    for item in raw_payload.get("mcp_servers", []):
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        mcp_servers.append(
            {
                "name": str(item.get("name", "MCP")).strip() or "MCP",
                "url": url,
                "api_key": str(item.get("api_key", "") or ""),
                "enabled": bool(item.get("enabled", True)),
                "timeout_ms": _normalized_mcp_timeout_ms(item.get("timeout_ms"), url=url),
            }
        )

    return {
        "generation": generation,
        "embeddings": embeddings,
        "reranker": reranker,
        "retrieval": retrieval,
        "mcp_servers": mcp_servers,
    }


def _redact_provider(provider_payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(provider_payload)
    has_secret = bool((provider_payload.get("api_key") or "").strip())
    redacted["api_key"] = "__configured__" if has_secret else ""
    return redacted


def _redact_mcp_server(item: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(item)
    has_secret = bool((item.get("api_key") or "").strip())
    redacted["api_key"] = "__configured__" if has_secret else ""
    return redacted


def _serialize_settings_for_output(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation": _redact_provider(payload["generation"]),
        "embeddings": _redact_provider(payload["embeddings"]),
        "reranker": _redact_provider(payload["reranker"]),
        "retrieval": payload.get("retrieval", {"chunk_count": 20}),
        "mcp_servers": [_redact_mcp_server(item) for item in payload.get("mcp_servers", [])],
    }


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
            api_key = ""
            if include_secrets and server.api_key_encrypted and encryption_service.enabled:
                try:
                    api_key_payload, _ = encryption_service.decrypt_json(server.api_key_encrypted)
                    api_key = str(api_key_payload.get("api_key", "") or "")
                except Exception as exc:
                    logger.warning("Failed to decrypt MCP server key for %s/%s: %s", user_id, server.id, exc)
            elif not include_secrets and server.api_key_encrypted:
                api_key = "__configured__"

            resolved.append(
                {
                    "name": server.name,
                    "url": server.url,
                    "api_key": api_key,
                    "enabled": bool(server.enabled),
                    "timeout_ms": _normalized_mcp_timeout_ms(server.timeout_ms, url=server.url),
                }
            )
        payload["mcp_servers"] = resolved

    return payload


def _runtime_options_from_settings(settings_payload: dict[str, Any]) -> AgentRuntimeOptions:
    settings = get_settings()
    generation = settings_payload.get("generation", {})
    retrieval = settings_payload.get("retrieval", {})
    mcp_servers = settings_payload.get("mcp_servers", [])
    enabled_mcp = [item for item in mcp_servers if bool(item.get("enabled", True))]
    mcp_endpoint_configs: list[dict[str, Any]] = []
    for item in enabled_mcp:
        endpoint = str(item.get("url", "")).strip()
        if not endpoint:
            continue
        mcp_endpoint_configs.append(
            {
                "url": endpoint,
                "api_key": str(item.get("api_key", "") or "").strip() or None,
                "timeout_ms": _normalized_mcp_timeout_ms(item.get("timeout_ms"), url=endpoint),
            }
        )

    resolved_openai_key = str(generation.get("api_key") or "").strip() or settings.openai_api_key
    resolved_openai_base = str(generation.get("base_url") or "").strip() or settings.openai_api_base
    resolved_llm_model = str(generation.get("model") or "").strip() or settings.llm_model
    resolved_mcp_endpoints = mcp_endpoint_configs or settings.default_mcp_endpoints or settings.resolved_mcp_endpoints()

    return AgentRuntimeOptions(
        openai_api_key=resolved_openai_key,
        openai_api_base=resolved_openai_base,
        llm_model=resolved_llm_model,
        rag_top_k=_normalized_chunk_count(retrieval.get("chunk_count"), default=20),
        rag_base_url=settings.rag_base_url,
        rag_query_path=settings.rag_query_path,
        mcp_endpoints=resolved_mcp_endpoints,
        mcp_api_key=settings.default_mcp_api_key or settings.mcp_api_key,
        mcp_rag_tool_name=settings.mcp_rag_tool_name,
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
    return {
        "id": message.id,
        "thread_id": message.thread_id,
        "role": message.role,
        "content": message.content,
        "reasoning_summary": message.reasoning_summary or "",
        "usage": message.usage_json or {},
        "citations": message.citations_json or [],
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def _stream_agent_response(
    *,
    prompt: str,
    thread_id: str | None,
    agent_builder: Callable[[], OntoPortalAgent],
    on_completed: Callable[[dict[str, Any], str], None] | None = None,
    log_context: dict[str, Any] | None = None,
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
            llm = _build_chat_model(runtime_options)
            resolved_model = str(runtime_options.llm_model or get_settings().llm_model or "")
            logger.info(_log_event("assistant_stream_mode", **stream_context, execution_mode="runtime", model=resolved_model))
            yield _sse({"type": "status", "message": "Classifying request..."})
            intent = _classify_intent(llm, prompt)
            logger.info(_log_event("assistant_stream_intent", **stream_context, intent=intent, model=resolved_model))
            if intent == "EDIT":
                yield _sse({"type": "status", "message": "Edit workflow uses buffered execution."})
                final_state = _collect_graph_final_state(agent=agent, prompt=prompt, thread_id=thread_id)
                for event in _emit_final_state(final_state):
                    yield event
                final_response_text = str(final_state.get("final_response") or final_state.get("rag_result") or "").strip()
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

                yield _sse({"type": "status", "message": "Streaming answer..."})
                messages = _build_response_messages(
                    question=prompt,
                    rag_result=str(final_state.get("rag_result") or ""),
                    citations=list(final_state.get("citations") or []),
                    retrieval_backend=str(final_state.get("retrieval_backend") or "unknown"),
                    retrieval_error=str(final_state.get("retrieval_error") or ""),
                )

                streamed_chunks: list[str] = []
                stream_usage: dict[str, Any] = {"model": resolved_model}
                try:
                    for chunk in llm.stream(messages):
                        text = _flatten_chunk_text(getattr(chunk, "content", chunk))
                        if not text:
                            continue
                        streamed_chunks.append(text)
                        yield _sse({"type": "delta", "content": text})
                except Exception as exc:  # pragma: no cover - fallback path
                    logger.warning(
                        _log_event(
                            "assistant_stream_chunk_fallback",
                            **stream_context,
                            intent=intent,
                            model=resolved_model,
                            **_failure_log_fields(exc),
                        )
                    )
                    reply = llm.invoke(messages)
                    stream_usage.update(_extract_generation_usage(reply))
                    text = _flatten_chunk_text(getattr(reply, "content", reply)).strip()
                    streamed_chunks = []
                    for chunk in _iter_text_chunks(text):
                        streamed_chunks.append(chunk)
                        yield _sse({"type": "delta", "content": chunk})

                if not streamed_chunks:
                    reply = llm.invoke(messages)
                    stream_usage.update(_extract_generation_usage(reply))
                    text = _flatten_chunk_text(getattr(reply, "content", reply)).strip()
                    streamed_chunks = []
                    for chunk in _iter_text_chunks(text):
                        streamed_chunks.append(chunk)
                        yield _sse({"type": "delta", "content": chunk})

                final_response_text = "".join(streamed_chunks).strip() or "No response generated."
                final_state["final_response"] = final_response_text
                final_state["generation_backend"] = f"llm:{resolved_model}"
                final_state["generation_usage"] = stream_usage

                reasoning_text = ""
                try:
                    summary_reply = llm.invoke(
                        _build_reasoning_messages(
                            question=prompt,
                            answer=final_response_text,
                            citations=list(final_state.get("citations") or []),
                        )
                    )
                    reasoning_text = _flatten_chunk_text(getattr(summary_reply, "content", summary_reply)).strip()
                    stream_usage.update(_extract_generation_usage(summary_reply))
                    model_reasoning = _extract_generation_reasoning(summary_reply)
                    if model_reasoning and not reasoning_text:
                        reasoning_text = model_reasoning
                except Exception as exc:  # pragma: no cover - summary is optional
                    logger.warning(
                        _log_event(
                            "assistant_reasoning_summary_failed",
                            **stream_context,
                            intent=intent,
                            model=resolved_model,
                            **_failure_log_fields(exc),
                        )
                    )

                if reasoning_text:
                    final_state["generation_reasoning"] = reasoning_text
                    yield _sse({"type": "model_reasoning", "content": reasoning_text})
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


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


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


@app.put("/api/v1/me/settings")
def me_put_settings(
    payload: AssistantSettingsIn,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    user_context = _resolve_user_context(request)
    encryption_service = _encryption_required()

    existing_payload = _load_effective_settings(session, user_id=user_context.user_id, include_secrets=True)
    normalized_payload = _normalize_settings_payload(payload.model_dump())
    for provider_key in ("generation", "embeddings", "reranker"):
        incoming_provider = normalized_payload.get(provider_key, {})
        existing_provider = existing_payload.get(provider_key, {})
        if not str(incoming_provider.get("api_key", "")).strip() and str(existing_provider.get("api_key", "")).strip():
            incoming_provider["api_key"] = str(existing_provider.get("api_key", "")).strip()

    existing_mcp_by_identity = {
        (item.get("name", "").strip(), item.get("url", "").strip()): item
        for item in existing_payload.get("mcp_servers", [])
    }
    for server in normalized_payload.get("mcp_servers", []):
        identity = (server.get("name", "").strip(), server.get("url", "").strip())
        existing_server = existing_mcp_by_identity.get(identity)
        if existing_server and not str(server.get("api_key", "")).strip() and str(existing_server.get("api_key", "")).strip():
            server["api_key"] = str(existing_server.get("api_key", "")).strip()

    settings_blob = {
        "generation": normalized_payload["generation"],
        "embeddings": normalized_payload["embeddings"],
        "reranker": normalized_payload["reranker"],
        "retrieval": normalized_payload["retrieval"],
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
        encrypt_api_key=lambda api_key: encryption_service.encrypt_json({"api_key": api_key}),
    )
    return _serialize_settings_for_output(normalized_payload)


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
            client = McpClient(
                [
                    {
                        "url": endpoint,
                        "api_key": item.get("api_key") or "",
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

    def on_completed(final_state: dict[str, Any], final_response_text: str) -> None:
        create_message(
            session,
            user_id=user_context.user_id,
            thread_id=thread.thread_id,
            role="assistant",
            content=final_response_text or "No response generated.",
            reasoning_summary=str(final_state.get("generation_reasoning") or ""),
            usage_json=final_state.get("generation_usage") if isinstance(final_state.get("generation_usage"), dict) else {},
            citations_json=final_state.get("citations") if isinstance(final_state.get("citations"), list) else [],
        )

    return StreamingResponse(
        _stream_agent_response(
            prompt=prompt,
            thread_id=thread.thread_id,
            agent_builder=lambda: OntoPortalAgent(runtime_options=runtime_options),
            on_completed=on_completed,
            log_context=log_context,
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
