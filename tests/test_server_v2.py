import importlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

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
                "model": "gemini-2.5-flash-lite",
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
    assert get_body["generation"]["model"] == "gemini-2.5-flash-lite"
    assert get_body["generation"]["api_key"] == "__configured__"
    assert get_body["retrieval"]["chunk_count"] == 12
    assert get_body["mcp_servers"][0]["api_key"] == "__configured__"


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
            llm_model="gemini-2.5-flash-lite",
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
                "model": "gemini-2.5-flash-lite",
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

    monkeypatch.setattr(server, "_build_chat_model", lambda _runtime_options: _DummyLlm())
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
            llm_model="gemini-2.5-flash-lite",
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
