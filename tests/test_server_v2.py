import json
import logging
import importlib
import time
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
from ontoportal_agent.db.user_context import build_signature


def _configure_env(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "assistant-v2-test.db"
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_DATABASE_URL", f"sqlite:///{db_path}")
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
    assert state["citations"] == ["PROCESSONTOLOGY v1.0"]
    assert observed["top_k"] == 12
    assert "mcp_called" not in observed


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


def test_stream_agent_response_runs_edit_flow_on_vertex_via_openai_bridge(monkeypatch):
    captured = {"graph_calls": [], "agent_runtime_options": []}

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

    def _collect_graph(agent, prompt, thread_id):
        captured["graph_calls"].append(
            {
                "agent": agent,
                "prompt": prompt,
                "thread_id": thread_id,
            }
        )
        return {
            "final_response": "Proposed ontology edits (pending approval):\n- Add a class",
            "generation_usage": {"model": "google/gemini-2.5-pro"},
        }

    class _BufferedEditAgent:
        def __init__(self, *args, runtime_options=None, **kwargs):
            captured["agent_runtime_options"].append(runtime_options)
            self.runtime_options = runtime_options

    monkeypatch.setattr(server, "_classify_intent", lambda _llm, _prompt: "EDIT")
    monkeypatch.setattr(server, "_vertex_access_token", lambda _runtime_options: "vertex-token-123")
    monkeypatch.setattr(server, "_collect_graph_final_state", _collect_graph)
    monkeypatch.setattr(server, "OntoPortalAgent", _BufferedEditAgent)

    events = "".join(
        server._stream_agent_response(
            prompt="Create a new ontology class for tensile strength.",
            thread_id="thread-edit-1",
            agent_builder=lambda: initial_agent,
        )
    )

    assert "Edit workflow uses buffered execution." in events
    assert "Proposed ontology edits (pending approval)" in events
    assert len(captured["graph_calls"]) == 1
    assert len(captured["agent_runtime_options"]) == 1
    bridged = captured["agent_runtime_options"][0]
    assert bridged.generation_provider == "openai_compatible"
    assert bridged.openai_api_key == "vertex-token-123"
    assert bridged.openai_api_base == "https://aiplatform.googleapis.com/v1/projects/ontoportal-llm-finetune/locations/global/endpoints/openapi"
    assert bridged.llm_model == "google/gemini-2.5-pro"


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
