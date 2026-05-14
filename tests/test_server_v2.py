import json
import logging
import importlib
import os
import subprocess
import time
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage, SystemMessage

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from fastapi.testclient import TestClient

from ontoportal_agent import config as config_module
from ontoportal_agent import server
from ontoportal_agent.db.base import get_engine, get_session_factory
from ontoportal_agent.db.repositories import create_message
from ontoportal_agent.db.user_context import build_signature
from ontoportal_agent.opencode_executor import OpenCodeExecutionResult


def _configure_env(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "assistant-v2-test.db"
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path / "ontology-workdir"))
    monkeypatch.setenv("ONTOAGENT_ENCRYPTION_KEY_CURRENT", "A" * 32)
    monkeypatch.setenv("ONTOAGENT_USER_CONTEXT_SECRET", "ctx-secret")
    monkeypatch.setenv("ONTOAGENT_INTERNAL_API_TOKEN", "internal-token")
    monkeypatch.setenv("ONTOAGENT_MCP_ENDPOINTS", "")

    config_module.get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    server._agent_instance = None
    server.startup()


def _signed_headers(
    *,
    user_id: str = "user-1",
    username: str = "alice",
    email: str = "alice@example.org",
    include_internal_token: bool = False,
) -> dict[str, str]:
    secret = config_module.get_settings().user_context_secret or ""
    timestamp = str(int(time.time()))
    headers = {
        "X-Assistant-User-Id": user_id,
        "X-Assistant-Username": username,
        "X-Assistant-User-Email": email,
        "X-Assistant-User-Timestamp": timestamp,
        "X-Assistant-User-Signature": build_signature(
            secret,
            user_id,
            username,
            email,
            timestamp,
        ),
    }
    if include_internal_token:
        headers["X-Internal-Token"] = config_module.get_settings().internal_api_token or ""
    return headers


def test_settings_crud_redacts_secrets(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()

    response = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "openai_compatible",
                "model": "gemini-3-flash-preview",
                "api_key": "user-gen-key",
                "base_url": "https://example.test/openai",
            },
            "embeddings": {
                "provider": "openai_compatible",
                "model": "text-embedding-005",
                "api_key": "user-embed-key",
                "base_url": "https://example.test/openai",
            },
            "reranker": {
                "provider": "cohere",
                "model": "rerank-v3.5",
                "api_key": "user-rerank-key",
                "base_url": "https://example.test/cohere",
            },
            "retrieval": {
                "chunk_count": 12,
            },
            "mcp_servers": [
                {
                    "name": "Primary RAG",
                    "url": "https://mcp.example.org",
                    "api_key": "mcp-user-key",
                    "enabled": True,
                    "timeout_ms": 12000,
                }
            ],
        },
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["generation"]["api_key"] == "__configured__"
    assert body["embeddings"]["api_key"] == "__configured__"
    assert body["reranker"]["api_key"] == "__configured__"
    assert body["retrieval"]["chunk_count"] == 12
    assert body["mcp_servers"][0]["api_key"] == "__configured__"

    get_response = client.get("/api/v1/me/settings", headers=headers)
    assert get_response.status_code == 200
    get_body = get_response.json()
    assert get_body["generation"]["model"] == "gemini-3-flash-preview"
    assert get_body["generation"]["api_key"] == "__configured__"
    assert get_body["retrieval"]["chunk_count"] == 12
    assert get_body["mcp_servers"][0]["api_key"] == "__configured__"

    masked_save = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "openai_compatible",
                "model": "gemini-2.5-pro",
                "api_key": "__configured__",
                "base_url": "https://example.test/openai",
            },
            "embeddings": get_body["embeddings"],
            "reranker": get_body["reranker"],
            "retrieval": {"chunk_count": 8},
            "mcp_servers": get_body["mcp_servers"],
        },
        headers=headers,
    )
    assert masked_save.status_code == 200
    masked_body = masked_save.json()
    assert masked_body["generation"]["api_key"] == "__configured__"
    assert masked_body["generation"]["model"] == "gemini-2.5-pro"
    assert masked_body["mcp_servers"][0]["api_key"] == "__configured__"


def test_switching_provider_does_not_preserve_previous_secret(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()

    initial = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "openai_compatible",
                "model": "gemini-3.1-pro-preview",
                "api_key": "old-openai-key",
                "base_url": "https://example.test/openai",
            },
            "embeddings": {
                "provider": "openai_compatible",
                "model": "text-embedding-005",
                "api_key": "",
                "base_url": "https://example.test/openai",
            },
            "reranker": {
                "provider": "none",
                "model": "",
                "api_key": "",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert initial.status_code == 200
    assert initial.json()["generation"]["api_key"] == "__configured__"

    switched = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "vertex_gemini",
                "model": "gemini-2.5-pro",
                "api_key": "",
                "base_url": "",
            },
            "embeddings": {
                "provider": "openai_compatible",
                "model": "text-embedding-005",
                "api_key": "",
                "base_url": "https://example.test/openai",
            },
            "reranker": {
                "provider": "none",
                "model": "",
                "api_key": "",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert switched.status_code == 200
    assert switched.json()["generation"]["api_key"] == ""

    repeated = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "vertex_gemini",
                "model": "gemini-2.5-pro",
                "api_key": "",
                "base_url": "",
            },
            "embeddings": {
                "provider": "openai_compatible",
                "model": "text-embedding-005",
                "api_key": "",
                "base_url": "https://example.test/openai",
            },
            "reranker": {
                "provider": "none",
                "model": "",
                "api_key": "",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert repeated.status_code == 200
    assert repeated.json()["generation"]["api_key"] == ""


def test_gemini_api_settings_preserve_user_api_key(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()

    saved = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-2.5-pro",
                "api_key": "user-gemini-api-key",
                "base_url": "",
            },
            "embeddings": {
                "provider": "openai_compatible",
                "model": "text-embedding-005",
                "api_key": "",
                "base_url": "https://example.test/openai",
            },
            "reranker": {
                "provider": "none",
                "model": "",
                "api_key": "",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert saved.status_code == 200
    saved_body = saved.json()
    assert saved_body["generation"]["api_key"] == "__configured__"
    assert saved_body["generation"]["provider"] == "gemini_api"
    assert saved_body["generation"]["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"

    repeated = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-3-flash-preview",
                "api_key": "__configured__",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            },
            "embeddings": saved_body["embeddings"],
            "reranker": saved_body["reranker"],
            "retrieval": saved_body["retrieval"],
            "mcp_servers": saved_body["mcp_servers"],
        },
        headers=headers,
    )
    assert repeated.status_code == 200
    repeated_body = repeated.json()
    assert repeated_body["generation"]["api_key"] == "__configured__"
    assert repeated_body["generation"]["model"] == "gemini-3-flash-preview"
    assert repeated_body["generation"]["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_settings_clear_generation_api_key(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()

    saved = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-2.5-pro",
                "api_key": "user-gemini-api-key",
                "base_url": "",
            },
            "embeddings": {"provider": "openai_compatible", "model": "", "api_key": "", "base_url": ""},
            "reranker": {"provider": "none", "model": "", "api_key": "", "base_url": ""},
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json()["generation"]["api_key"] == "__configured__"

    cleared = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-2.5-pro",
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "clear_api_key": True,
            },
            "embeddings": saved.json()["embeddings"],
            "reranker": saved.json()["reranker"],
            "retrieval": saved.json()["retrieval"],
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert cleared.status_code == 200
    assert cleared.json()["generation"]["api_key"] == ""

    loaded = client.get("/api/v1/me/settings", headers=headers)
    assert loaded.status_code == 200
    assert loaded.json()["generation"]["api_key"] == ""


def test_settings_persist_opencode_auth_source(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()

    saved = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-2.5-pro",
                "api_key": "",
                "base_url": "",
            },
            "embeddings": {"provider": "openai_compatible", "model": "", "api_key": "", "base_url": ""},
            "reranker": {"provider": "none", "model": "", "api_key": "", "base_url": ""},
            "retrieval": {"chunk_count": 12},
            "opencode": {"auth_source": "opencode_builtin"},
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert saved.status_code == 200
    assert saved.json()["opencode"] == {"auth_source": "opencode_builtin"}

    loaded = client.get("/api/v1/me/settings", headers=headers)
    assert loaded.status_code == 200
    assert loaded.json()["opencode"] == {"auth_source": "opencode_builtin"}


def test_provider_check_uses_saved_gemini_api_key_without_returning_secret(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()
    seen = {}

    class FakeResponse:
        status_code = 200
        content = b'{"data":[{"id":"gemini-2.5-pro"}]}'
        text = content.decode("utf-8")

        def json(self):
            return {"data": [{"id": "gemini-2.5-pro"}]}

    def fake_get(url, *, headers=None, timeout=None):
        seen["url"] = url
        seen["authorization"] = (headers or {}).get("Authorization")
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(server.requests, "get", fake_get)

    saved = client.put(
        "/api/v1/me/settings",
        json={
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-2.5-pro",
                "api_key": "user-gemini-api-key",
                "base_url": "",
            },
            "embeddings": {"provider": "openai_compatible", "model": "", "api_key": "", "base_url": ""},
            "reranker": {"provider": "none", "model": "", "api_key": "", "base_url": ""},
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        },
        headers=headers,
    )
    assert saved.status_code == 200

    checked = client.post(
        "/api/v1/me/settings/provider/check",
        json={
            "scope": "generation",
            "provider": "gemini_api",
            "model": "gemini-2.5-pro",
            "api_key": "",
            "base_url": "",
        },
        headers=headers,
    )
    assert checked.status_code == 200
    body = checked.json()
    assert body["ok"] is True
    assert body["model_available"] is True
    assert "api_key" not in body
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/models"
    assert seen["authorization"] == "Bearer user-gemini-api-key"


def test_provider_check_redacts_failed_key(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()
    fake_gemini_key = "AI" + "za" + "123456789012345678901234567890"

    class FakeResponse:
        status_code = 401
        content = f'{{"error":"bad key {fake_gemini_key}"}}'.encode("utf-8")
        text = content.decode("utf-8")

        def json(self):
            return {"error": "bad key"}

    monkeypatch.setattr(server.requests, "get", lambda *args, **kwargs: FakeResponse())

    checked = client.post(
        "/api/v1/me/settings/provider/check",
        json={
            "scope": "generation",
            "provider": "gemini_api",
            "model": "gemini-2.5-pro",
            "api_key": fake_gemini_key,
            "base_url": "",
        },
        headers=headers,
    )
    assert checked.status_code == 400
    detail = checked.json()["detail"]
    assert fake_gemini_key not in detail
    assert "[redacted]" in detail


def test_provider_check_redacts_bearer_and_openai_style_keys(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()
    fake_openai_key = "sk" + "-testsecret1234567890"

    class FakeResponse:
        status_code = 401
        content = f"Authorization: Bearer {fake_openai_key} api_key={fake_openai_key}".encode("utf-8")
        text = content.decode("utf-8")

        def json(self):
            return {"error": "bad key"}

    monkeypatch.setattr(server.requests, "get", lambda *args, **kwargs: FakeResponse())

    checked = client.post(
        "/api/v1/me/settings/provider/check",
        json={
            "scope": "generation",
            "provider": "openai_compatible",
            "model": "gpt-test",
            "api_key": fake_openai_key,
            "base_url": "https://example.test/v1",
        },
        headers=headers,
    )
    assert checked.status_code == 400
    detail = checked.json()["detail"]
    assert fake_openai_key not in detail
    assert "Bearer [redacted]" in detail
    assert "api_key=[redacted]" in detail


def test_serialize_message_hides_non_displayable_reasoning():
    hidden = server._serialize_message(
        SimpleNamespace(
            id=1,
            thread_id="thread-1",
            role="assistant",
            content="Answer.",
            reasoning_summary="Synthetic explanation.",
            usage_json={"reasoning_kind": "provider_thought_stream"},
            citations_json=[],
            created_at=None,
        )
    )
    shown = server._serialize_message(
        SimpleNamespace(
            id=2,
            thread_id="thread-1",
            role="assistant",
            content="Answer.",
            reasoning_summary="Native provider thought stream.",
            usage_json={
                "reasoning_kind": "provider_thought_stream",
                "reasoning_displayable": True,
            },
            citations_json=[],
            created_at=None,
        )
    )

    assert hidden["reasoning_summary"] == ""
    assert shown["reasoning_summary"] == "Native provider thought stream."


def test_me_chat_stream_persists_messages(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)

    dummy_state = {
        "generation_reasoning": "- chose closest ontology chunk",
        "generation_usage": {"model": "test-model", "total_tokens": 11},
        "citations": ["MDS v1"],
        "final_response": "Synthetic assistant answer.",
    }

    def _stream(_payload, config=None, stream_mode=None):
        assert stream_mode == "updates"
        yield {"classify": {"intent": "RETRIEVE"}}
        yield {"respond": dummy_state}

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            self.graph = SimpleNamespace(stream=_stream)

    monkeypatch.setattr(server, "OntoPortalAgent", _DummyAgent)

    client = TestClient(server.app)
    headers = _signed_headers(include_internal_token=True)

    thread_resp = client.post("/api/v1/me/threads", json={"title": "Test thread"}, headers=headers)
    assert thread_resp.status_code == 200
    thread_id = thread_resp.json()["thread_id"]

    response = client.post(
        "/api/v1/me/chat/stream",
        json={"prompt": "What is MDS?", "thread_id": thread_id},
        headers=headers,
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "[DONE]" in response.text

    messages_resp = client.get(f"/api/v1/me/threads/{thread_id}/messages", headers=headers)
    assert messages_resp.status_code == 200
    messages = messages_resp.json()["messages"]
    assert [msg["role"] for msg in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "What is MDS?"
    assert "Synthetic assistant answer." in messages[1]["content"]


def test_me_chat_stream_exposes_and_persists_quota_errors(monkeypatch, tmp_path, caplog):
    _configure_env(monkeypatch, tmp_path)

    class _QuotaError(Exception):
        status_code = 429

    class _DummyLlm:
        def stream(self, _messages):
            raise _QuotaError("Error code: 429 - quota exceeded. Please retry in 9.1s.")

        def invoke(self, _messages):
            raise _QuotaError("Error code: 429 - quota exceeded. Please retry in 9.1s.")

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            self.runtime_options = SimpleNamespace(
                openai_api_key="test-openai-key",
                openai_api_base="https://example.test/openai",
                llm_model="gemini-3-flash-preview",
                rag_top_k=12,
                rag_base_url="http://rag.internal",
                rag_query_path="/api/v1/query",
                mcp_endpoints=[],
                mcp_api_key=None,
                mcp_rag_tool_name="rag_query",
            )

    monkeypatch.setattr(server, "OntoPortalAgent", _DummyAgent)
    monkeypatch.setattr(server, "_build_chat_model", lambda _runtime_options, model_override=None: _DummyLlm())
    monkeypatch.setattr(server, "_classify_intent", lambda _llm, _prompt: "RETRIEVE")
    monkeypatch.setattr(
        server,
        "_retrieve_runtime_state",
        lambda _prompt, _runtime_options: {
            "citations": [],
            "rag_result": "",
            "retrieval_backend": "rag-http",
            "retrieval_error": "",
            "retrieval_chunk_count": 12,
        },
    )

    client = TestClient(server.app)
    headers = _signed_headers(include_internal_token=True)

    thread_resp = client.post("/api/v1/me/threads", json={"title": "Quota thread"}, headers=headers)
    assert thread_resp.status_code == 200
    thread_id = thread_resp.json()["thread_id"]

    with caplog.at_level(logging.INFO):
        response = client.post(
            "/api/v1/me/chat/stream",
            json={"prompt": "Follow up question", "thread_id": thread_id},
            headers=headers,
        )

    assert response.status_code == 200
    events = [
        json.loads(line.replace("data:", "", 1).strip())
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    error_event = next(event for event in events if event.get("type") == "error")
    assert error_event["content"]["status"] == "Provider quota exceeded."
    assert error_event["content"]["status_code"] == 429
    assert error_event["content"]["retry_after_seconds"] == 10
    assert "Add your own API key in AI Settings" in response.text

    messages_resp = client.get(f"/api/v1/me/threads/{thread_id}/messages", headers=headers)
    assert messages_resp.status_code == 200
    messages = messages_resp.json()["messages"]
    assert [msg["role"] for msg in messages] == ["user", "assistant"]
    assert "configured AI provider quota is exhausted" in messages[1]["content"]
    assert messages[1]["usage"]["error"]["status_code"] == 429
    assert messages[1]["usage"]["error"]["trace_id"]

    assert any("assistant_stream_failed" in record.message and "trace_id=" in record.message for record in caplog.records)


def test_me_chat_stream_retries_same_reasoning_model_before_fallback(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)

    class _TransientError(Exception):
        status_code = 503

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            self.runtime_options = SimpleNamespace(
                openai_api_key="test-openai-key",
                openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
                llm_model="gemini-3.1-pro-preview-customtools",
                rag_top_k=12,
                rag_base_url="http://rag.internal",
                rag_query_path="/api/v1/query",
                mcp_endpoints=[],
                mcp_api_key=None,
                mcp_rag_tool_name="rag_query",
            )

    attempts: list[str] = []

    def _stream_events(*, model, usage_state, answer_chunks, reasoning_chunks, **_kwargs):
        attempts.append(model)
        if len(attempts) == 1:
            raise _TransientError("Error code: 503 - temporarily unavailable")
        reasoning_chunks.append("Provider reasoning.")
        usage_state["reasoning_kind"] = "provider_thought_stream"
        yield {"type": "reasoning_delta", "content": "Provider reasoning."}
        answer_chunks.append("Recovered answer.")
        yield {"type": "delta", "content": "Recovered answer."}

    monkeypatch.setattr(server, "OntoPortalAgent", _DummyAgent)
    monkeypatch.setattr(server, "_build_chat_model", lambda _runtime_options, model_override=None: SimpleNamespace())
    monkeypatch.setattr(server, "_classify_intent", lambda _llm, _prompt: "RETRIEVE")
    monkeypatch.setattr(
        server,
        "_retrieve_runtime_state",
        lambda _prompt, _runtime_options: {
            "citations": [],
            "rag_result": "",
            "retrieval_backend": "rag-http",
            "retrieval_error": "",
            "retrieval_chunk_count": 12,
        },
    )
    monkeypatch.setattr(server, "_stream_openai_compatible_events", _stream_events)

    client = TestClient(server.app)
    headers = _signed_headers(include_internal_token=True)

    thread_resp = client.post("/api/v1/me/threads", json={"title": "Retry thread"}, headers=headers)
    assert thread_resp.status_code == 200
    thread_id = thread_resp.json()["thread_id"]

    response = client.post(
        "/api/v1/me/chat/stream",
        json={"prompt": "Explain interoperability", "thread_id": thread_id},
        headers=headers,
    )

    assert response.status_code == 200
    assert attempts == [
        "gemini-3.1-pro-preview-customtools",
        "gemini-3.1-pro-preview-customtools",
    ]
    assert "Retrying gemini-3.1-pro-preview-customtools after a transient provider failure." in response.text
    assert "\"type\": \"reasoning_delta\"" in response.text
    assert "Recovered answer." in response.text
    assert "Switching to gemini-3.1-pro-preview" not in response.text


def test_me_chat_stream_resets_stale_reasoning_before_fallback(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)

    class _TransientError(Exception):
        status_code = 503

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            self.runtime_options = SimpleNamespace(
                openai_api_key="test-openai-key",
                openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
                llm_model="gemini-3.1-pro-preview-customtools",
                rag_top_k=12,
                rag_base_url="http://rag.internal",
                rag_query_path="/api/v1/query",
                mcp_endpoints=[],
                mcp_api_key=None,
                mcp_rag_tool_name="rag_query",
            )

    attempts: list[str] = []

    def _stream_events(*, model, usage_state, answer_chunks, reasoning_chunks, **_kwargs):
        attempts.append(model)
        if model == "gemini-3.1-pro-preview-customtools":
            reasoning_chunks.append("Stale reasoning.")
            yield {"type": "reasoning_delta", "content": "Stale reasoning."}
            raise _TransientError("Error code: 503 - temporarily unavailable")
        reasoning_chunks.append("Fresh reasoning.")
        usage_state["reasoning_kind"] = "provider_thought_stream"
        yield {"type": "reasoning_delta", "content": "Fresh reasoning."}
        answer_chunks.append("Fallback answer.")
        yield {"type": "delta", "content": "Fallback answer."}

    monkeypatch.setattr(server, "OntoPortalAgent", _DummyAgent)
    monkeypatch.setattr(server, "_build_chat_model", lambda _runtime_options, model_override=None: SimpleNamespace())
    monkeypatch.setattr(server, "_classify_intent", lambda _llm, _prompt: "RETRIEVE")
    monkeypatch.setattr(
        server,
        "_retrieve_runtime_state",
        lambda _prompt, _runtime_options: {
            "citations": [],
            "rag_result": "",
            "retrieval_backend": "rag-http",
            "retrieval_error": "",
            "retrieval_chunk_count": 12,
        },
    )
    monkeypatch.setattr(server, "_stream_openai_compatible_events", _stream_events)

    client = TestClient(server.app)
    headers = _signed_headers(include_internal_token=True)

    thread_resp = client.post("/api/v1/me/threads", json={"title": "Fallback thread"}, headers=headers)
    assert thread_resp.status_code == 200
    thread_id = thread_resp.json()["thread_id"]

    response = client.post(
        "/api/v1/me/chat/stream",
        json={"prompt": "Explain interoperability", "thread_id": thread_id},
        headers=headers,
    )

    assert response.status_code == 200
    assert attempts == [
        "gemini-3.1-pro-preview-customtools",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3.1-pro-preview",
    ]
    assert "\"type\": \"reasoning_reset\"" in response.text
    assert "Switching to gemini-3.1-pro-preview." in response.text
    assert "Fallback answer." in response.text


def test_stream_vertex_gemini_events_emits_reasoning_and_usage(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"candidates":[{"content":{"parts":[{"text":"Thinking step.","thought":true}]}}],"usageMetadata":{"trafficType":"ON_DEMAND"},"modelVersion":"gemini-2.5-pro"}'
            yield 'data: {"candidates":[{"content":{"parts":[{"text":"Final answer."}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"totalTokenCount":30,"thoughtsTokenCount":16},"modelVersion":"gemini-2.5-pro"}'

    captured = {}

    def _fake_post(url, headers, json, stream, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["stream"] = stream
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(server, "_vertex_access_token", lambda _runtime_options: "token-123")
    monkeypatch.setattr(server, "_vertex_endpoint_url", lambda _runtime_options, model: f"https://vertex.test/{model}")
    monkeypatch.setattr(server.requests, "post", _fake_post)

    usage = {"model": "gemini-2.5-pro"}
    answer_chunks = []
    reasoning_chunks = []
    runtime_options = SimpleNamespace(generation_provider="vertex_gemini")

    events = list(
        server._stream_vertex_gemini_events(
            runtime_options=runtime_options,
            model="gemini-2.5-pro",
            messages=[
                SystemMessage(content="System guidance."),
                HumanMessage(content="Explain ontology interoperability."),
            ],
            usage_state=usage,
            answer_chunks=answer_chunks,
            reasoning_chunks=reasoning_chunks,
        )
    )

    assert events == [
        {"type": "reasoning_delta", "content": "Thinking step."},
        {"type": "delta", "content": "Final answer."},
    ]
    assert usage["model"] == "gemini-2.5-pro"
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 30
    assert usage["reasoning_tokens"] == 16
    assert usage["reasoning_kind"] == "provider_thought_stream"
    assert usage["reasoning_displayable"] is True
    assert answer_chunks == ["Final answer."]
    assert reasoning_chunks == ["Thinking step."]
    assert captured["url"] == "https://vertex.test/gemini-2.5-pro"
    assert captured["headers"]["Authorization"] == "Bearer token-123"
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == "System guidance."
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "Explain ontology interoperability."
    assert captured["stream"] is True


def test_retrieve_runtime_state_prefers_direct_rag(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            mcp_api_key="shared-key",
            mcp_rag_tool_name="rag_query",
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )

    observed: dict[str, object] = {}

    class _DummyRagClient:
        def __init__(self, *, base_url=None, query_path=None):
            observed["base_url"] = base_url
            observed["query_path"] = query_path

        def query(self, prompt, *, top_k=None):
            observed["prompt"] = prompt
            observed["top_k"] = top_k
            return SimpleNamespace(
                answer="Direct RAG answer",
                sources=[SimpleNamespace(ontology_id="PROCESSONTOLOGY", version="1.0")],
            )

    class _DummyMcpClient:
        def __init__(self, *args, **kwargs):
            observed["mcp_init"] = True

        def invoke_rag_query(self, *args, **kwargs):
            observed["mcp_called"] = True
            raise AssertionError("MCP should not be used when direct RAG is available")

    monkeypatch.setattr(server, "RagClient", _DummyRagClient)
    monkeypatch.setattr(server, "McpClient", _DummyMcpClient)

    state = server._retrieve_runtime_state(
        "Summarize PROCESSONTOLOGY",
        SimpleNamespace(
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            rag_top_k=12,
            mcp_endpoints=[{"url": "http://rag.internal/mcp", "timeout_ms": 10000}],
            mcp_api_key="shared-key",
            mcp_rag_tool_name="rag_query",
        ),
    )

    assert state["retrieval_backend"] == "rag-http"
    assert state["rag_result"] == "Direct RAG answer"
    assert state["citations"][0]["document_label"] == "PROCESSONTOLOGY v1.0"
    assert state["citations"][0]["ontology_id"] == "PROCESSONTOLOGY"
    assert observed["top_k"] == 12
    assert "mcp_called" not in observed


def test_artifact_endpoints_list_view_download_and_bundle(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()

    thread_resp = client.post("/api/v1/me/threads", json={"title": "Artifacts"}, headers=headers)
    assert thread_resp.status_code == 200
    thread_id = thread_resp.json()["thread_id"]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess_result = subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True, text=True)
    assert subprocess_result.returncode == 0
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.org"], cwd=str(workspace), check=True)
    (workspace / "proposal.ttl").write_text("@prefix ex: <https://example.org/> .\n", encoding="utf-8")
    (workspace / "notes.md").write_text("# Notes\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(workspace), check=True)

    run_id = "run-artifacts-1"
    usage = {
        "mode": "opencode",
        "execution": {
            "mode": "opencode",
            "ok": True,
            "run_id": run_id,
            "workspace": str(workspace),
            "changed_files": [
                {"status": "A", "path": "proposal.ttl", "kind": "ttl"},
                {"status": "A", "path": "notes.md", "kind": "md"},
            ],
            "artifact_candidates": [{"status": "A", "path": "proposal.ttl", "kind": "ttl"}],
            "diff_summary": {},
            "expires_at": "2999-01-01T00:00:00+00:00",
        },
    }
    with get_session_factory()() as session:
        create_message(
            session,
            user_id="user-1",
            thread_id=thread_id,
            role="assistant",
            content="OpenCode finished.",
            usage_json=usage,
            citations_json=[],
        )

    files_resp = client.get(f"/api/v1/me/artifacts/{thread_id}/{run_id}/files", headers=headers)
    assert files_resp.status_code == 200
    files = files_resp.json()["files"]
    assert [item["path"] for item in files] == ["proposal.ttl", "notes.md"]
    assert files[0]["viewable"] is True
    assert "absolute_path" not in files[0]

    view_resp = client.get(
        f"/api/v1/me/artifacts/{thread_id}/{run_id}/file",
        params={"path": "proposal.ttl"},
        headers=headers,
    )
    assert view_resp.status_code == 200
    assert view_resp.json()["language"] == "turtle"
    assert "@prefix ex:" in view_resp.json()["content"]

    diff_resp = client.get(
        f"/api/v1/me/artifacts/{thread_id}/{run_id}/file",
        params={"path": "proposal.ttl", "view": "diff"},
        headers=headers,
    )
    assert diff_resp.status_code == 200
    assert "proposal.ttl" in diff_resp.json()["content"]

    download_resp = client.get(
        f"/api/v1/me/artifacts/{thread_id}/{run_id}/download",
        params={"path": "notes.md"},
        headers=headers,
    )
    assert download_resp.status_code == 200
    assert download_resp.content == b"# Notes\n"

    bundle_resp = client.get(f"/api/v1/me/artifacts/{thread_id}/{run_id}/bundle.zip", headers=headers)
    assert bundle_resp.status_code == 200
    with zipfile.ZipFile(BytesIO(bundle_resp.content)) as archive:
        assert sorted(archive.namelist()) == ["notes.md", "proposal.ttl"]


def test_artifact_endpoints_enforce_owner_and_safe_paths(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()
    other_headers = _signed_headers(user_id="user-2", username="bob", email="bob@example.org")

    thread_resp = client.post("/api/v1/me/threads", json={"title": "Safe artifacts"}, headers=headers)
    assert thread_resp.status_code == 200
    thread_id = thread_resp.json()["thread_id"]

    workspace = tmp_path / "workspace-safe"
    workspace.mkdir()
    (workspace / "ok.ttl").write_text("ok", encoding="utf-8")
    (workspace / "opencode.json").write_text('{"api_key":"secret"}', encoding="utf-8")
    outside = tmp_path / "outside.ttl"
    outside.write_text("outside", encoding="utf-8")
    (workspace / "leak.ttl").symlink_to(outside)
    run_id = "run-safe-1"
    with get_session_factory()() as session:
        create_message(
            session,
            user_id="user-1",
            thread_id=thread_id,
            role="assistant",
            content="OpenCode finished.",
            usage_json={
                "execution": {
                    "run_id": run_id,
                    "workspace": str(workspace),
                    "changed_files": [
                        {"status": "A", "path": "ok.ttl", "kind": "ttl"},
                        {"status": "A", "path": "leak.ttl", "kind": "ttl"},
                    ],
                    "artifact_candidates": [],
                    "expires_at": "2999-01-01T00:00:00+00:00",
                }
            },
            citations_json=[],
        )

    unsafe_resp = client.get(
        f"/api/v1/me/artifacts/{thread_id}/{run_id}/file",
        params={"path": "../outside.ttl"},
        headers=headers,
    )
    assert unsafe_resp.status_code == 400

    unlisted_resp = client.get(
        f"/api/v1/me/artifacts/{thread_id}/{run_id}/file",
        params={"path": "opencode.json"},
        headers=headers,
    )
    assert unlisted_resp.status_code == 404

    symlink_resp = client.get(
        f"/api/v1/me/artifacts/{thread_id}/{run_id}/file",
        params={"path": "leak.ttl"},
        headers=headers,
    )
    assert symlink_resp.status_code == 400

    other_resp = client.get(f"/api/v1/me/artifacts/{thread_id}/{run_id}/files", headers=other_headers)
    assert other_resp.status_code == 404


def test_admin_artifact_cleanup_requires_internal_token_and_removes_expired_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENCODE_ARTIFACT_RETENTION_DAYS", "1")
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    settings = config_module.get_settings()
    root = settings.ontology_workdir / settings.opencode_workspace_subdir
    root.mkdir(parents=True, exist_ok=True)
    old_workspace = root / "thread-old-run"
    fresh_workspace = root / "thread-fresh-run"
    for workspace in (old_workspace, fresh_workspace):
        workspace.mkdir()
        (workspace / "opencode.json").write_text("{}", encoding="utf-8")
    old_time = time.time() - (3 * 24 * 60 * 60)
    os.utime(old_workspace, (old_time, old_time))

    forbidden = client.post("/api/v1/admin/artifacts/cleanup")
    assert forbidden.status_code == 403

    response = client.post(
        "/api/v1/admin/artifacts/cleanup",
        headers={"X-Internal-Token": settings.internal_api_token or ""},
    )

    assert response.status_code == 200
    assert response.json()["removed_workspaces"] == 1
    assert not old_workspace.exists()
    assert fresh_workspace.exists()


def test_builtin_mcp_timeout_is_upgraded_for_runtime(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="test-openai-key",
            openai_api_base="https://example.test/openai",
            llm_model="gemini-3-flash-preview",
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "openai_compatible",
                "model": "gemini-3-flash-preview",
                "api_key": "",
                "base_url": "https://example.test/openai",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [
                {
                    "name": "MCP 1",
                    "url": "http://rag.internal/mcp",
                    "api_key": "",
                    "enabled": True,
                    "timeout_ms": 10000,
                }
            ],
        }
    )

    assert runtime_options.mcp_endpoints == [
        {
            "url": "http://rag.internal/mcp",
            "api_key": None,
            "timeout_ms": 30000,
        }
    ]


def test_runtime_options_tracks_user_generation_key_for_opencode(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="deployment-openai-key",
            openai_api_base="https://deployment.example/openai",
            llm_model="deployment-model",
            default_generation_provider="openai_compatible",
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "openai_compatible",
                "model": "gpt-5.2",
                "api_key": "user-openai-key",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        }
    )

    assert runtime_options.openai_api_key == "user-openai-key"
    assert runtime_options.generation_api_key_configured is True

    auth = server._opencode_provider_auth_from_runtime_options(runtime_options)
    assert auth is not None
    assert auth.model_ref == "matportal-user/gpt-5.2"
    assert auth.api_key == "user-openai-key"
    assert auth.base_url == "https://deployment.example/openai"


def test_runtime_options_treat_gemini_api_key_as_openai_compatible_for_opencode(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="deployment-openai-key",
            openai_api_base="https://deployment.example/openai",
            llm_model="deployment-model",
            default_generation_provider="openai_compatible",
            vertex_project=None,
            vertex_location="us-central1",
            vertex_service_account_json=None,
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-2.5-pro",
                "api_key": "user-gemini-api-key",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        }
    )

    assert runtime_options.generation_provider == "openai_compatible"
    assert runtime_options.openai_api_key == "user-gemini-api-key"
    assert runtime_options.openai_api_base == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert runtime_options.generation_api_key_configured is True

    auth = server._opencode_provider_auth_from_runtime_options(runtime_options)
    assert auth is not None
    assert auth.model_ref == "matportal-user/gemini-2.5-pro"
    assert auth.api_key == "user-gemini-api-key"
    assert auth.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_opencode_builtin_auth_source_skips_user_generation_key(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="deployment-openai-key",
            openai_api_base="https://deployment.example/openai",
            llm_model="deployment-model",
            default_generation_provider="openai_compatible",
            vertex_project=None,
            vertex_location="us-central1",
            vertex_service_account_json=None,
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "gemini_api",
                "model": "gemini-2.5-pro",
                "api_key": "user-gemini-api-key",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "opencode": {"auth_source": "opencode_builtin"},
            "mcp_servers": [],
        }
    )

    assert runtime_options.opencode_auth_source == "opencode_builtin"
    assert runtime_options.generation_api_key_configured is True
    assert server._opencode_provider_auth_from_runtime_options(runtime_options) is None


def test_opencode_usage_reports_auth_source():
    result = OpenCodeExecutionResult(
        ok=True,
        workspace="/tmp/workspace",
        run_id="run-auth",
        expires_at="2999-01-01T00:00:00+00:00",
        model="opencode/big-pickle",
    )
    runtime_options = SimpleNamespace(
        opencode_auth_source="generation_key",
        generation_api_key_configured=True,
    )

    payload = server._opencode_usage_payload(result, runtime_options)

    assert payload["execution"]["auth_source"] == "generation_key"
    assert payload["execution"]["using_user_generation_key"] is True


def test_runtime_options_track_user_vertex_account_for_opencode(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="deployment-openai-key",
            openai_api_base="https://deployment.example/openai",
            llm_model="deployment-model",
            default_generation_provider="openai_compatible",
            vertex_project="deployment-project",
            vertex_location="us-central1",
            vertex_service_account_json='{"project_id":"deployment-project"}',
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )
    service_account_json = json.dumps(
        {
            "type": "service_account",
            "project_id": "json-user-project",
            "client_email": "svc@example.org",
            "private_key": "-----BEGIN " + "PRIVATE KEY-----\nabc\n-----END " + "PRIVATE KEY-----\n",
        }
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "vertex_gemini",
                "model": "gemini-2.5-pro",
                "api_key": service_account_json,
                "base_url": "",
                "project": "explicit-user-project",
                "location": "europe-west4",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        }
    )

    assert runtime_options.generation_provider == "vertex_gemini"
    assert runtime_options.generation_api_key_configured is True
    assert runtime_options.vertex_service_account_json == service_account_json
    assert runtime_options.vertex_project == "explicit-user-project"
    assert runtime_options.vertex_location == "europe-west4"
    assert runtime_options.openai_api_key == "deployment-openai-key"

    monkeypatch.setattr(server, "_vertex_access_token", lambda _runtime_options: "vertex-access-token")
    auth = server._opencode_provider_auth_from_runtime_options(runtime_options)
    assert auth is not None
    assert auth.model_ref == "matportal-user/google/gemini-2.5-pro"
    assert auth.api_key == "vertex-access-token"
    assert auth.base_url == (
        "https://aiplatform.googleapis.com/v1/projects/explicit-user-project/locations/global/endpoints/openapi"
    )


def test_runtime_options_use_vertex_project_from_user_service_account(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="deployment-openai-key",
            openai_api_base="https://deployment.example/openai",
            llm_model="deployment-model",
            default_generation_provider="vertex_gemini",
            vertex_project="deployment-project",
            vertex_location="us-central1",
            vertex_service_account_json=None,
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )
    service_account_json = json.dumps(
        {
            "type": "service_account",
            "project_id": "json-user-project",
            "client_email": "svc@example.org",
            "private_key": "-----BEGIN " + "PRIVATE KEY-----\nabc\n-----END " + "PRIVATE KEY-----\n",
        }
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "vertex_gemini",
                "model": "gemini-2.5-pro",
                "api_key": service_account_json,
                "base_url": "",
                "project": "",
                "location": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        }
    )

    assert runtime_options.vertex_project == "json-user-project"
    assert runtime_options.vertex_location == "us-central1"


def test_opencode_auth_skips_deployment_default_key(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="deployment-openai-key",
            openai_api_base="https://deployment.example/openai",
            llm_model="deployment-model",
            default_generation_provider="openai_compatible",
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "openai_compatible",
                "model": "gpt-5.2",
                "api_key": "",
                "base_url": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        }
    )

    assert runtime_options.openai_api_key == "deployment-openai-key"
    assert runtime_options.generation_api_key_configured is False
    assert server._opencode_provider_auth_from_runtime_options(runtime_options) is None


def test_opencode_auth_skips_deployment_default_vertex_account(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key="deployment-openai-key",
            openai_api_base="https://deployment.example/openai",
            llm_model="deployment-model",
            default_generation_provider="vertex_gemini",
            vertex_project="deployment-project",
            vertex_location="us-central1",
            vertex_service_account_json='{"project_id":"deployment-project"}',
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            default_mcp_endpoints=[],
            default_mcp_api_key=None,
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
            resolved_mcp_endpoints=lambda: ["http://rag.internal/mcp"],
        ),
    )

    runtime_options = server._runtime_options_from_settings(
        {
            "generation": {
                "provider": "vertex_gemini",
                "model": "gemini-2.5-pro",
                "api_key": "",
                "base_url": "",
                "project": "",
                "location": "",
            },
            "retrieval": {"chunk_count": 12},
            "mcp_servers": [],
        }
    )

    assert runtime_options.generation_api_key_configured is False
    assert runtime_options.vertex_service_account_json == '{"project_id":"deployment-project"}'
    assert server._opencode_provider_auth_from_runtime_options(runtime_options) is None


def test_me_routes_reject_bad_signature(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = _signed_headers()
    headers["X-Assistant-User-Signature"] = "invalid"

    response = client.get("/api/v1/me/bootstrap", headers=headers)
    assert response.status_code == 401


def test_stream_agent_response_keeps_markdown_queries_in_retrieve_mode(monkeypatch):
    observed = {"graph_called": False}

    class _DummyLlm:
        def stream(self, _messages):
            yield AIMessage(content="# Summary\n")
            yield AIMessage(content="- MatPortal helps organize ontology knowledge.\n")

        def invoke(self, messages):
            system_text = str(getattr(messages[0], "content", ""))
            if "Route the user request" in system_text:
                return AIMessage(content="EDIT")
            if "Write a concise reasoning summary" in system_text:
                return AIMessage(content="- Used the retrieved context.\n- Kept the answer concise.")
            return AIMessage(content="Fallback answer")

    def _unexpected_edit_graph(*args, **kwargs):
        observed["graph_called"] = True
        raise AssertionError("retrieve-style prompts must not enter the edit graph")

    monkeypatch.setattr(server, "_build_chat_model", lambda _runtime_options, model_override=None: _DummyLlm())
    monkeypatch.setattr(
        server,
        "_retrieve_runtime_state",
        lambda prompt, runtime_options: {
            "rag_result": "MatPortal is an ontology portal for materials science.",
            "citations": ["MATONTO v2.0"],
            "retrieval_backend": "rag-http",
            "retrieval_error": "",
        },
    )
    monkeypatch.setattr(server, "_collect_graph_final_state", _unexpected_edit_graph)

    runtime_agent = SimpleNamespace(
        runtime_options=SimpleNamespace(
            llm_model="gemini-3-flash-preview",
            openai_api_key="test-key",
            openai_api_base="https://example.test/openai",
        )
    )

    events = "".join(
        server._stream_agent_response(
            prompt="What is MatPortal? Answer in markdown with a heading, twelve bullet points, and a short json code block.",
            thread_id="thread-123",
            agent_builder=lambda: runtime_agent,
        )
    )

    assert observed["graph_called"] is False
    assert "Streaming answer..." in events
    assert "No action generated; placeholder plan." not in events
    assert "# Summary" in events


def test_stream_agent_response_routes_edit_prompts_to_opencode_workspace(monkeypatch):
    initial_agent = SimpleNamespace(
        runtime_options=SimpleNamespace(
            generation_provider="vertex_gemini",
            llm_model="gemini-2.5-pro",
            openai_api_key="",
            openai_api_base="",
            vertex_project="ontoportal-llm-finetune",
            vertex_location="us-central1",
            vertex_service_account_json='{"client_email":"svc@example.org"}',
        )
    )

    def _fake_opencode_stream(*, prompt, thread_id, trace_id, runtime_options):
        assert "tensile strength" in prompt
        assert thread_id == "thread-edit-1"
        assert trace_id
        assert runtime_options is initial_agent.runtime_options
        yield server._sse(
            {
                "type": "workspace_mode",
                "content": {
                    "mode": "execution",
                    "run_id": "run-test-1",
                    "workspace": "/tmp/ontoportal-agent/opencode-runs/thread-edit-1",
                },
            }
        )
        yield server._sse({"type": "terminal_log", "content": {"line": "[bash] ls -la"}})
        yield server._sse({"type": "changed_files", "content": [{"status": "A", "path": "proposal.ttl"}]})
        yield server._sse({"type": "artifact_candidates", "content": [{"path": "proposal.ttl"}]})
        return OpenCodeExecutionResult(
            ok=True,
            workspace="/tmp/ontoportal-agent/opencode-runs/thread-edit-1",
            run_id="run-test-1",
            expires_at="2999-01-01T00:00:00+00:00",
            model="opencode/big-pickle",
            final_text="Prepared a Turtle proposal for review.",
            changed_files=[{"status": "A", "path": "proposal.ttl"}],
            artifact_candidates=[{"path": "proposal.ttl"}],
            diff_summary={"stat": "1 file changed"},
        )

    monkeypatch.setattr(server, "_classify_intent", lambda _llm, _prompt: "EDIT")
    monkeypatch.setattr(server, "_stream_opencode_execution", _fake_opencode_stream)

    events = "".join(
        server._stream_agent_response(
            prompt="Create a new ontology class for tensile strength.",
            thread_id="thread-edit-1",
            agent_builder=lambda: initial_agent,
        )
    )

    assert "Prepared a Turtle proposal for review." in events
    assert '"type": "changed_files"' in events
    assert '"run_id": "run-test-1"' in events
    assert '"mode": "opencode"' in events


def test_stream_agent_response_respects_requested_edit_mode(monkeypatch):
    initial_agent = SimpleNamespace(
        runtime_options=SimpleNamespace(
            generation_provider="vertex_gemini",
            llm_model="gemini-2.5-pro",
            openai_api_key="",
            openai_api_base="",
            vertex_project="ontoportal-llm-finetune",
            vertex_location="us-central1",
            vertex_service_account_json='{"client_email":"svc@example.org"}',
        )
    )

    def _unexpected_classifier(_llm, _prompt):
        raise AssertionError("explicit edit mode should not call the classifier")

    def _fake_opencode_stream(*, prompt, thread_id, trace_id, runtime_options):
        assert "Summarize the current proposal" in prompt
        assert thread_id == "thread-edit-mode"
        assert trace_id
        assert runtime_options is initial_agent.runtime_options
        yield server._sse(
            {
                "type": "workspace_mode",
                "content": {
                    "mode": "execution",
                    "run_id": "run-forced-edit",
                    "workspace": "/tmp/ontoportal-agent/opencode-runs/thread-edit-mode",
                },
            }
        )
        return OpenCodeExecutionResult(
            ok=True,
            workspace="/tmp/ontoportal-agent/opencode-runs/thread-edit-mode",
            run_id="run-forced-edit",
            expires_at="2999-01-01T00:00:00+00:00",
            model="opencode/big-pickle",
            final_text="Ran the request in the OpenCode workspace.",
        )

    monkeypatch.setattr(server, "_classify_intent", _unexpected_classifier)
    monkeypatch.setattr(server, "_stream_opencode_execution", _fake_opencode_stream)

    events = "".join(
        server._stream_agent_response(
            prompt="Summarize the current proposal in notes.md.",
            thread_id="thread-edit-mode",
            agent_builder=lambda: initial_agent,
            requested_mode="edit",
        )
    )

    assert "Starting OpenCode workspace..." in events
    assert "Ran the request in the OpenCode workspace." in events
    assert '"run_id": "run-forced-edit"' in events


def test_stream_agent_response_hybrid_ask_uses_opencode_after_backend_retrieval(monkeypatch):
    runtime_agent = SimpleNamespace(
        runtime_options=SimpleNamespace(
            generation_provider="openai_compatible",
            llm_model="gpt-5.2",
            openai_api_key="test-key",
            openai_api_base="https://api.openai.com/v1",
            rag_top_k=12,
            rag_base_url="http://rag.internal",
            rag_query_path="/api/v1/query",
            mcp_endpoints=[],
            mcp_api_key=None,
            mcp_rag_tool_name="rag_query",
        )
    )
    observed: dict[str, object] = {}

    def _fake_opencode_ask_stream(*, prompt, thread_id, trace_id, runtime_options, retrieval_state):
        observed["prompt"] = prompt
        observed["thread_id"] = thread_id
        observed["trace_id"] = trace_id
        observed["runtime_options"] = runtime_options
        observed["retrieval_state"] = retrieval_state
        yield server._sse({"type": "status", "message": "OpenCode ask generation running."})
        return OpenCodeExecutionResult(
            ok=True,
            workspace="/tmp/ontoportal-agent/opencode-runs/thread-hybrid-ask",
            run_id="run-hybrid-ask",
            expires_at="2999-01-01T00:00:00+00:00",
            model="opencode/big-pickle",
            final_text="OpenCode generated the answer from MATONTO context.",
        )

    monkeypatch.setattr(server, "_build_chat_model", lambda _runtime_options, model_override=None: SimpleNamespace())
    monkeypatch.setattr(server, "_classify_intent", lambda _llm, _prompt: "RETRIEVE")
    monkeypatch.setattr(server, "_opencode_hybrid_ask_enabled", lambda: True)
    monkeypatch.setattr(server, "_stream_opencode_ask_generation", _fake_opencode_ask_stream)
    monkeypatch.setattr(
        server,
        "_retrieve_runtime_state",
        lambda prompt, runtime_options: {
            "rag_result": "MATONTO covers materials terminology.",
            "citations": [{"document_label": "MATONTO v2.0", "rank": 1}],
            "citation_labels": ["MATONTO v2.0"],
            "retrieval_backend": "rag-http",
            "retrieval_error": "",
            "retrieval_chunk_count": 12,
        },
    )
    monkeypatch.setattr(
        server,
        "_stream_openai_compatible_events",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("standard LLM path should not run")),
    )

    events = "".join(
        server._stream_agent_response(
            prompt="Which ontology should I use for aluminium?",
            thread_id="thread-hybrid-ask",
            agent_builder=lambda: runtime_agent,
        )
    )

    assert observed["thread_id"] == "thread-hybrid-ask"
    assert observed["runtime_options"] is runtime_agent.runtime_options
    assert observed["retrieval_state"]["rag_result"] == "MATONTO covers materials terminology."
    assert "OpenCode ask generation running." in events
    assert "OpenCode generated the answer from MATONTO context." in events
    assert '"type": "citations"' in events
    assert '"document_label": "MATONTO v2.0"' in events
    assert '"mode": "opencode_hybrid_ask"' in events
    assert "Streaming answer..." not in events


def test_stream_agent_response_falls_back_to_latest_google_model(monkeypatch):
    class _BusyError(Exception):
        status_code = 503

    class _BusyLlm:
        def stream(self, _messages):
            raise _BusyError("Error code: 503 - This model is currently experiencing high demand.")

        def invoke(self, _messages):
            raise _BusyError("Error code: 503 - This model is currently experiencing high demand.")

    def _build_llm(runtime_options, *, model_override=None):
        model = model_override or runtime_options.llm_model
        return SimpleNamespace(model=model)

    def _stream_openai_events(*, runtime_options, model, messages, usage_state, answer_chunks, reasoning_chunks):
        if model == "gemini-3.1-pro-preview":
            raise _BusyError("Error code: 503 - This model is currently experiencing high demand.")
        if model == "gemini-3.1-pro-preview-customtools":
            raise _BusyError("Error code: 503 - This model is currently experiencing high demand.")
        if model == "gemini-3-pro-preview":
            usage_state["model"] = model
            reasoning_chunks.append("Fallback reasoning from Gemini 3 Pro.")
            usage_state["reasoning_kind"] = "provider_thought_stream"
            yield {"type": "reasoning_delta", "content": "Fallback reasoning from Gemini 3 Pro."}
            answer_chunks.append("Fallback answer from Gemini 3 Pro.")
            yield {"type": "delta", "content": "Fallback answer from Gemini 3 Pro."}
            return
        raise AssertionError(f"unexpected model {model}")

    monkeypatch.setattr(server, "_build_chat_model", _build_llm)
    monkeypatch.setattr(server, "_stream_openai_compatible_events", _stream_openai_events)
    monkeypatch.setattr(server, "_classify_intent", lambda _llm, _prompt: "RETRIEVE")
    monkeypatch.setattr(
        server,
        "_retrieve_runtime_state",
        lambda prompt, runtime_options: {
            "rag_result": "MatPortal context",
            "citations": ["MATONTO v2.0"],
            "retrieval_backend": "rag-http",
            "retrieval_error": "",
        },
    )

    runtime_agent = SimpleNamespace(
        runtime_options=SimpleNamespace(
            llm_model="gemini-3.1-pro-preview",
            openai_api_key="test-key",
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        )
    )

    events = "".join(
        server._stream_agent_response(
            prompt="Explain MatPortal.",
            thread_id="thread-456",
            agent_builder=lambda: runtime_agent,
        )
    )

    assert "Primary model unavailable. Switching to gemini-3-pro-preview." in events
    assert "Fallback reasoning from Gemini 3 Pro." in events
    assert "Fallback answer from Gemini 3 Pro." in events
