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
        _env_file=None,
    )
    assert settings.openai_api_key == "test"
    assert settings.ontoportal_api_key == "key"
    assert settings.ontology_workdir == tmp_path
    assert settings.mcp_endpoints == ["http://mcp.example.com/mcp", "http://other/mcp"]
    assert settings.resolved_mcp_endpoints() == ["http://mcp.example.com/mcp", "http://other/mcp"]
