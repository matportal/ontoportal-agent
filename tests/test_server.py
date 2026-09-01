import importlib
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from fastapi.testclient import TestClient

from ontoportal_agent import server


@pytest.mark.parametrize(
    ("configured_token", "request_token"),
    [("secret-token", None), ("secret-token", "wrong-token"), (None, "anything")],
)
def test_chat_stream_requires_configured_matching_internal_token(monkeypatch, configured_token, request_token):
    monkeypatch.setattr(server, "get_settings", lambda: SimpleNamespace(internal_api_token=configured_token))
    client = TestClient(server.app)
    headers = {"X-Internal-Token": request_token} if request_token is not None else {}

    response = client.post(
        "/api/v1/chat/stream",
        json={"prompt": "What is aluminium?", "thread_id": "thread-1"},
        headers=headers,
    )
    assert response.status_code == 403


def test_chat_stream_emits_sse_payload(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(internal_api_token="secret-token", encryption_key_current="A" * 32, encryption_key_previous=None),
    )
    dummy_state = {
        "generation_reasoning": "- Picked BWMD because it has explicit Aluminium classes.",
        "generation_usage": {
            "model": "test-model",
            "reasoning_kind": "provider_thought_stream",
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "reasoning_tokens": 7,
            "total_tokens": 46,
        },
        "citations": ["TEST v1"],
        "final_response": "Aluminium is a metallic element.",
    }
    seen = {}

    def stream(payload, config=None, stream_mode=None):
        seen["payload"] = payload
        seen["config"] = config
        seen["stream_mode"] = stream_mode
        yield {"classify": {"intent": "RETRIEVE"}}
        yield {"retrieve": {"retrieval_backend": "mcp"}}
        yield {"respond": dummy_state}

    dummy_agent = SimpleNamespace(graph=SimpleNamespace(stream=stream))
    monkeypatch.setattr(server, "_get_agent", lambda: dummy_agent)

    client = TestClient(server.app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"prompt": "What is aluminium?", "thread_id": "thread-2"},
        headers={"X-Internal-Token": "secret-token"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "Assistant request received" in response.text
    assert "\"type\": \"reasoning_delta\"" not in response.text
    assert "Picked BWMD because it has explicit Aluminium classes." not in response.text
    assert "\"type\": \"usage\"" in response.text
    assert "\"model\": \"test-model\"" in response.text
    assert "\"reasoning_tokens\": 7" in response.text
    assert "Aluminium is a metallic element." in response.text
    assert "[DONE]" in response.text
    assert seen["payload"] == {"user_input": "What is aluminium?"}
    assert seen["config"] == {"configurable": {"thread_id": "thread-2"}}
    assert seen["stream_mode"] == "updates"
