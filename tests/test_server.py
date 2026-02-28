import importlib
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from fastapi.testclient import TestClient

from ontoportal_agent import server


def test_chat_stream_requires_internal_token_when_configured(monkeypatch):
    monkeypatch.setattr(server, "get_settings", lambda: SimpleNamespace(internal_api_token="secret-token"))
    client = TestClient(server.app)

    response = client.post(
        "/api/v1/chat/stream",
        json={"prompt": "What is aluminium?", "thread_id": "thread-1"},
    )
    assert response.status_code == 403


def test_chat_stream_emits_sse_payload(monkeypatch):
    monkeypatch.setattr(server, "get_settings", lambda: SimpleNamespace(internal_api_token=None))
    dummy_state = {
        "retrieval_backend": "mcp",
        "generation_backend": "llm:test-model",
        "generation_usage": {
            "model": "test-model",
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "reasoning_tokens": 7,
            "total_tokens": 46,
        },
        "citations": ["TEST v1"],
        "final_response": "Aluminium is a metallic element.",
    }
    seen = {}

    def invoke(payload, config=None):
        seen["payload"] = payload
        seen["config"] = config
        return dummy_state

    dummy_agent = SimpleNamespace(graph=SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(server, "_get_agent", lambda: dummy_agent)

    client = TestClient(server.app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"prompt": "What is aluminium?", "thread_id": "thread-2"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "Assistant request received" in response.text
    assert "retrieval_backend=mcp" in response.text
    assert "generation_backend=llm:test-model" in response.text
    assert "generation_model=test-model" in response.text
    assert "generation_prompt_tokens=12" in response.text
    assert "generation_completion_tokens=34" in response.text
    assert "generation_reasoning_tokens=7" in response.text
    assert "generation_total_tokens=46" in response.text
    assert "Aluminium is a metallic element." in response.text
    assert "[DONE]" in response.text
    assert seen["payload"] == {"user_input": "What is aluminium?"}
    assert seen["config"] == {"configurable": {"thread_id": "thread-2"}}
