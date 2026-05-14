import importlib

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

if importlib.util.find_spec("langgraph") is None:
    pytest.skip("langgraph dependency not installed", allow_module_level=True)

from ontoportal_agent.config import AgentSettings


def test_settings_defaults(tmp_path):
    settings = AgentSettings(
        OPENAI_API_KEY="test",
        ONTOPORTAL_API_KEY="key",
        ONTOLOGY_WORKDIR=str(tmp_path),
        MCP_ENDPOINTS=["http://mcp.example.com/mcp", "http://other/mcp"],
        MCP_API_KEY="mcp-secret",
        MCP_RAG_TOOL_NAME="search_ontology_knowledge",
        _env_file=None,
    )
    assert settings.openai_api_key == "test"
    assert settings.ontoportal_api_key == "key"
    assert settings.ontology_workdir == tmp_path
    assert settings.mcp_endpoints == ["http://mcp.example.com/mcp", "http://other/mcp"]
    assert settings.mcp_api_key == "mcp-secret"
    assert settings.mcp_rag_tool_name == "search_ontology_knowledge"
    assert settings.opencode_hybrid_ask_enabled is False
    assert settings.opencode_run_timeout_seconds == 900
    assert settings.resolved_mcp_endpoints() == ["http://mcp.example.com/mcp", "http://other/mcp"]


def test_settings_parses_comma_separated_mcp_endpoints(tmp_path):
    settings = AgentSettings(
        OPENAI_API_KEY="test",
        ONTOPORTAL_API_KEY="key",
        ONTOLOGY_WORKDIR=str(tmp_path),
        MCP_ENDPOINTS="http://mcp-a.example.com/mcp,http://mcp-b.example.com/mcp",
        _env_file=None,
    )
    assert settings.mcp_endpoints == ["http://mcp-a.example.com/mcp", "http://mcp-b.example.com/mcp"]


def test_settings_support_local_opencode_mcp_mode(tmp_path):
    settings = AgentSettings(
        OPENAI_API_KEY="test",
        ONTOPORTAL_API_KEY="key",
        ONTOLOGY_WORKDIR=str(tmp_path),
        OPENCODE_MCP_MODE="local",
        OPENCODE_MCP_SERVER_ROOT="/opt/ontoportal-api-mcp",
        OPENCODE_MCP_PYTHON="/venv/bin/python",
        _env_file=None,
    )
    assert settings.opencode_mcp_mode == "local"
    assert str(settings.opencode_mcp_server_root) == "/opt/ontoportal-api-mcp"
    assert settings.opencode_mcp_python == "/venv/bin/python"
