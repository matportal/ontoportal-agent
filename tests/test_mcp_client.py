import importlib

import pytest
import requests

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.mcp_client import McpClient, McpInvocationError


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_mcp_client_passes_api_key_header(monkeypatch):
    captured = {}

    def _fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response(
            {
                "tools": [
                    {
                        "name": "rag_query",
                        "description": "Query ontology content",
                        "arguments": {"query": {"type": "string"}},
                    }
                ]
            }
        )

    monkeypatch.setattr("ontoportal_agent.mcp_client.requests.get", _fake_get)

    client = McpClient(["http://rag.internal/mcp"], api_key="shared-secret")
    tools = client.list_tools()

    assert captured["url"] == "http://rag.internal/mcp/tools"
    assert captured["headers"] == {"X-API-Key": "shared-secret"}
    assert len(tools) == 1
    assert tools[0][1].name == "rag_query"


def test_invoke_rag_query_uses_invoke_endpoint(monkeypatch):
    posted = {}

    def _fake_get(url, headers, timeout):
        return _Response({"tools": [{"name": "rag_query", "description": "", "arguments": {}}]})

    def _fake_post(url, json, headers, timeout):
        posted["url"] = url
        posted["json"] = json
        posted["headers"] = headers
        return _Response({"answer": "Aluminium details", "sources": []})

    monkeypatch.setattr("ontoportal_agent.mcp_client.requests.get", _fake_get)
    monkeypatch.setattr("ontoportal_agent.mcp_client.requests.post", _fake_post)

    client = McpClient(["http://rag.internal/mcp"], api_key="shared-secret")
    payload = client.invoke_rag_query("aluminium")

    assert posted["url"] == "http://rag.internal/mcp/invoke"
    assert posted["json"] == {"tool": "rag_query", "arguments": {"query": "aluminium"}}
    assert posted["headers"] == {"X-API-Key": "shared-secret"}
    assert payload["answer"] == "Aluminium details"


def test_list_tools_wraps_transport_errors(monkeypatch):
    def _raise_request_error(url, headers, timeout):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr("ontoportal_agent.mcp_client.requests.get", _raise_request_error)

    client = McpClient(["http://rag.internal/mcp"], api_key="shared-secret")

    with pytest.raises(McpInvocationError):
        client.list_tools()


def test_mcp_client_uses_endpoint_specific_key_and_timeout(monkeypatch):
    captured = {}

    def _fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response({"tools": []})

    monkeypatch.setattr("ontoportal_agent.mcp_client.requests.get", _fake_get)

    client = McpClient(
        [{"url": "http://rag.internal/mcp", "api_key": "endpoint-key", "timeout_ms": 45000}],
        api_key="shared-secret",
    )
    client.list_tools()

    assert captured["url"] == "http://rag.internal/mcp/tools"
    assert captured["headers"] == {"X-API-Key": "endpoint-key"}
    assert captured["timeout"] == 45
