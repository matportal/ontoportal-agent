import importlib
import sys
from types import ModuleType

import pytest
from langchain_core.messages import AIMessage

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.config import get_settings
from ontoportal_agent.edit_runtime import DeepAgentsEditRuntime, OpenCodeEditRuntime, create_edit_runtime, normalize_edit_runtime_name
from ontoportal_agent.edit_runtime.base import EditRuntimeRequest


def test_normalize_edit_runtime_name_accepts_compatibility_aliases():
    assert normalize_edit_runtime_name(None) == "opencode"
    assert normalize_edit_runtime_name("open-code") == "opencode"
    assert normalize_edit_runtime_name("workspace") == "opencode"
    assert normalize_edit_runtime_name("deep-agents") == "deepagents"
    assert normalize_edit_runtime_name("langchain_deepagents") == "deepagents"
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


def test_create_edit_runtime_gates_deepagents(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="Deep Agents edit runtime is disabled"):
        create_edit_runtime("deepagents")

    monkeypatch.setenv("ONTOAGENT_DEEPAGENTS_ENABLED", "true")
    get_settings.cache_clear()
    runtime = create_edit_runtime("deepagents", model=object())

    assert isinstance(runtime, DeepAgentsEditRuntime)
    assert runtime.capabilities.runtime == "deepagents"
    assert runtime.capabilities.supports_artifacts is True


def test_deepagents_runtime_writes_artifacts_and_reuses_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_DEEPAGENTS_ENABLED", "true")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_ROBOT_ENABLED", "false")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_KEEP_WORKSPACE", "false")
    get_settings.cache_clear()

    captured: dict[str, object] = {}

    class _StateBackend:
        pass

    class _FilesystemPermission:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeDeepAgent:
        def __init__(self, tools):
            self.tools = {tool.name: tool for tool in tools}

        def invoke(self, payload):
            captured["payload"] = payload
            self.tools["matportal_write_artifact"].invoke(
                {
                    "path": "proposal.ttl",
                    "content": "@prefix ex: <https://example.org/> .\nex:Material a ex:Class .\n",
                }
            )
            self.tools["matportal_write_artifact"].invoke(
                {"path": "validation-summary.json", "content": '{"status":"passed"}\n'}
            )
            validation = self.tools["matportal_validate_workspace"].invoke({})
            captured["validation"] = validation
            self.tools["matportal_write_artifact"].invoke(
                {"path": "operator-report.md", "content": "# Operator report\n\nValidation ran before this file.\n"}
            )
            with pytest.raises(Exception):
                self.tools["matportal_write_artifact"].invoke(
                    {"path": "leaky-note.md", "content": "api_key = sk-testsecretvalue12345\n"}
                )
            return {"messages": [AIMessage(content="Deep Agents prepared proposal.ttl.")]}

    def _create_deep_agent(**kwargs):
        captured["kwargs"] = kwargs
        return _FakeDeepAgent(kwargs["tools"])

    deepagents_module = ModuleType("deepagents")
    deepagents_module.create_deep_agent = _create_deep_agent
    backends_module = ModuleType("deepagents.backends")
    backends_module.StateBackend = _StateBackend
    middleware_module = ModuleType("deepagents.middleware")
    middleware_module.FilesystemPermission = _FilesystemPermission
    monkeypatch.setitem(sys.modules, "deepagents", deepagents_module)
    monkeypatch.setitem(sys.modules, "deepagents.backends", backends_module)
    monkeypatch.setitem(sys.modules, "deepagents.middleware", middleware_module)

    runtime = DeepAgentsEditRuntime(model=object())
    events = list(
        runtime.stream(
            EditRuntimeRequest(
                prompt="Draft a test ontology artifact.",
                thread_id="thread-deepagents",
                trace_id="trace-deepagents",
            )
        )
    )

    # Exhausting list(runtime.stream(...)) discards the generator return value, so
    # assert through streamed parity events and the workspace diff contract.
    assert any(event["type"] == "workspace_mode" and event["content"].get("runtime") == "deepagents" for event in events)
    changed = next(event["content"] for event in events if event["type"] == "changed_files")
    changed_paths = {item["path"] for item in changed}
    assert changed_paths >= {"proposal.ttl", "validation-summary.json", "operator-report.md"}
    assert "leaky-note.md" not in changed_paths
    validation_event = next(event["content"] for event in events if event["type"] == "validation_report")
    assert validation_event["ok"] is True
    assert any(item["path"] == "proposal.ttl" and item["status"] == "passed" for item in validation_event["checked_files"])
    assert captured["kwargs"]["name"] == "matportal-deepagents-edit"
