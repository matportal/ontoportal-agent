import importlib

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.config import get_settings
from ontoportal_agent.edit_runtime import OpenCodeEditRuntime, create_edit_runtime, normalize_edit_runtime_name


def test_normalize_edit_runtime_name_accepts_compatibility_aliases():
    assert normalize_edit_runtime_name(None) == "opencode"
    assert normalize_edit_runtime_name("open-code") == "opencode"
    assert normalize_edit_runtime_name("workspace") == "opencode"
    assert normalize_edit_runtime_name("pi.dev") == "pi"


@pytest.mark.parametrize("name", ["unknown", "codex"])
def test_normalize_edit_runtime_name_rejects_unknown_runtime(name):
    with pytest.raises(ValueError):
        normalize_edit_runtime_name(name)


def test_create_edit_runtime_defaults_to_opencode(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    runtime = create_edit_runtime()

    assert isinstance(runtime, OpenCodeEditRuntime)
    assert runtime.capabilities.runtime == "opencode"
    assert runtime.capabilities.supports_mcp is True


def test_create_edit_runtime_rejects_pi_until_adapter_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="Pi edit runtime is not available"):
        create_edit_runtime("pi")
