import importlib
import json
import sys
from types import ModuleType

import pytest
from langchain_core.messages import AIMessage

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.config import get_settings
from ontoportal_agent.edit_runtime import DeepAgentsEditRuntime, OpenCodeEditRuntime, PiEditRuntime, create_edit_runtime, normalize_edit_runtime_name
from ontoportal_agent.edit_runtime import deepagents as deepagents_runtime
from ontoportal_agent.edit_runtime.base import EditRuntimeRequest
from ontoportal_agent.opencode_executor import OpenCodeAccountAuth


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


def test_create_edit_runtime_gates_pi(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="Pi edit runtime is disabled"):
        create_edit_runtime("pi")

    monkeypatch.setenv("ONTOAGENT_PI_ADAPTER_ENABLED", "true")
    monkeypatch.setenv("ONTOAGENT_PI_MODEL", "antigravity/gemini-3.5-flash")
    get_settings.cache_clear()
    runtime = create_edit_runtime("pi")

    assert isinstance(runtime, PiEditRuntime)
    assert runtime.capabilities.runtime == "pi"
    assert runtime.capabilities.supports_artifacts is True


def test_pi_runtime_writes_structured_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_PI_ADAPTER_ENABLED", "true")
    monkeypatch.setenv("ONTOAGENT_PI_PATH", "/usr/bin/true")
    monkeypatch.setenv("ONTOAGENT_PI_MODEL", "antigravity/gemini-3.5-flash")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_ROBOT_ENABLED", "false")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_KEEP_WORKSPACE", "false")
    get_settings.cache_clear()

    import ontoportal_agent.edit_runtime.pi as pi_runtime_module

    captured: dict[str, object] = {}
    real_run = pi_runtime_module.subprocess.run

    class _Completed:
        returncode = 0
        stderr = ""

        @property
        def stdout(self):
            payload = {
                "summary": "Pi prepared proposal artifacts.",
                "artifacts": [
                    {
                        "path": "proposal.ttl",
                        "content": "@prefix ex: <https://example.org/> .\nex:PiMaterial a ex:Class .\n",
                    },
                    {"path": "validation-summary.json", "content": '{"status":"passed"}\n'},
                ],
            }
            return "\n".join(
                [
                    '{"type":"session","version":3}',
                    json_line(
                        {
                            "type": "agent_end",
                            "messages": [
                                {"role": "assistant", "content": [{"type": "text", "text": json_dumps(payload)}]}
                            ],
                        }
                    ),
                ]
            )

    def json_dumps(value):
        import json

        return json.dumps(value)

    def json_line(value):
        import json

        return json.dumps(value)

    def _fake_run(command, **kwargs):
        if command and command[0] == "/usr/bin/true":
            captured["command"] = command
            captured["kwargs"] = kwargs
            return _Completed()
        return real_run(command, **kwargs)

    monkeypatch.setattr(pi_runtime_module.subprocess, "run", _fake_run)

    runtime = create_edit_runtime("pi")
    events = list(
        runtime.stream(
            EditRuntimeRequest(
                prompt="Draft a tiny Pi ontology artifact.",
                thread_id="thread-pi",
                trace_id="trace-pi",
            )
        )
    )

    assert captured["command"][:5] == ["/usr/bin/true", "--mode", "json", "--print", "--no-tools"]
    assert "--no-session" in captured["command"]
    assert any(event["type"] == "workspace_mode" and event["content"].get("runtime") == "pi" for event in events)
    changed = next(event["content"] for event in events if event["type"] == "changed_files")
    changed_paths = {item["path"] for item in changed}
    assert changed_paths >= {"proposal.ttl", "validation-summary.json"}
    validation_event = next(event["content"] for event in events if event["type"] == "validation_report")
    assert validation_event["ok"] is True


def test_pi_runtime_extracts_fenced_json_with_nested_artifact_fences(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    payload = {
        "summary": "Nested JSON artifact content is preserved.",
        "artifacts": [
            {
                "path": "operator-report.md",
                "content": "# Report\n\n```json\n{\n  \"status\": \"passed\"\n}\n```\n",
            },
            {"path": "validation-summary.json", "content": '{"status":"passed"}\n'},
        ],
    }

    parsed = PiEditRuntime()._extract_json_object(f"```json\n{json.dumps(payload)}\n```")

    assert parsed == payload


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


def test_deepagents_runtime_preserves_opencode_codex_account_auth_users(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_DEEPAGENTS_ENABLED", "true")
    get_settings.cache_clear()

    runtime = create_edit_runtime(
        "deepagents",
        account_auth=OpenCodeAccountAuth(
            kind="codex",
            opencode_auth_json='{"provider":"antigravity","refresh":"rotated-refresh"}',
            codex_auth_json='{"tokens":{"access_token":"codex-token"}}',
        ),
    )

    assert isinstance(runtime, OpenCodeEditRuntime)
    assert runtime.capabilities.runtime == "opencode"


def test_pi_runtime_preserves_opencode_account_auth_users(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_PI_ADAPTER_ENABLED", "true")
    get_settings.cache_clear()

    runtime = create_edit_runtime(
        "pi",
        account_auth=OpenCodeAccountAuth(
            kind="codex",
            codex_auth_json='{"tokens":{"access_token":"codex-token"}}',
        ),
    )

    assert isinstance(runtime, OpenCodeEditRuntime)
    assert runtime.capabilities.runtime == "opencode"


def test_deepagents_runtime_uses_antigravity_account_bridge(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_DEEPAGENTS_ENABLED", "true")
    get_settings.cache_clear()

    account_auth = OpenCodeAccountAuth(
        kind="gemini_antigravity",
        opencode_auth_json='{"google":{"refresh":"rotated-refresh","access":"access-token"}}',
        model_ref="google/antigravity-gemini-3-pro",
    )
    runtime = create_edit_runtime("deepagents", account_auth=account_auth)

    assert isinstance(runtime, DeepAgentsEditRuntime)
    assert runtime.account_auth is account_auth
    assert runtime._antigravity_proxy_model_id() == "gemini-3.1-pro-high"


def test_deepagents_runtime_starts_antigravity_proxy_from_saved_account(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_DEEPAGENTS_ENABLED", "true")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_KEEP_WORKSPACE", "false")
    monkeypatch.setenv("ONTOAGENT_ASK_RUNTIME_MODEL", "gemini-3.1-flash-lite-preview")
    get_settings.cache_clear()

    captured: dict[str, object] = {"models": []}
    original_popen = deepagents_runtime.subprocess.Popen

    class _FakeProcess:
        def __init__(self, args, cwd=None, env=None, **kwargs):
            if not args or args[0] != "antigravity-claude-proxy":
                self._real = original_popen(args, cwd=cwd, env=env, **kwargs)
                return
            self._real = None
            captured["args"] = args
            captured["cwd"] = cwd
            captured["env"] = env
            accounts_path = tmp_path.__class__(env["HOME"]) / ".config" / "antigravity-proxy" / "accounts.json"
            captured["accounts_config"] = json.loads(accounts_path.read_text(encoding="utf-8"))
            self.terminated = False

        def __getattr__(self, name):
            if self._real is not None:
                return getattr(self._real, name)
            raise AttributeError(name)

        def __enter__(self):
            return self._real.__enter__() if self._real is not None else self

        def __exit__(self, exc_type, exc, tb):
            if self._real is not None:
                return self._real.__exit__(exc_type, exc, tb)
            return False

        def poll(self):
            return self._real.poll() if self._real is not None else None

        def terminate(self):
            if self._real is not None:
                return self._real.terminate()
            captured["terminated"] = True
            self.terminated = True

        def wait(self, timeout=None):
            return self._real.wait(timeout=timeout) if self._real is not None else 0

    class _FakeResponse:
        status_code = 200

    class _FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured["chat_kwargs"] = kwargs

        def invoke(self, messages):
            captured["messages"] = messages
            return AIMessage(content="Antigravity account-auth answer.")

    monkeypatch.setattr(deepagents_runtime.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(deepagents_runtime.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(deepagents_runtime, "ChatAnthropic", _FakeChatAnthropic)
    monkeypatch.setattr(DeepAgentsEditRuntime, "_port_is_available", lambda self, port: True)

    account_auth = OpenCodeAccountAuth(
        kind="gemini_antigravity",
        opencode_auth_json=json.dumps(
            {
                "google": {
                    "refresh": "rotated-refresh|user-project",
                    "access": "access-token-should-not-be-written",
                    "email": "user@example.test",
                    "projectId": "user-project",
                }
            }
        ),
        model_ref="google/antigravity-gemini-3-pro",
    )
    runtime = DeepAgentsEditRuntime(account_auth=account_auth)
    events = list(
        runtime.stream(
            EditRuntimeRequest(
                prompt="What is MATONTO?",
                thread_id="thread-deepagents-account-ask",
                trace_id="trace-deepagents-account-ask",
                task="ask",
                retrieved_context="MATONTO context.",
            )
        )
    )

    assert captured["args"] == ["antigravity-claude-proxy", "start"]
    assert captured["env"]["HOST"] == "127.0.0.1"
    assert captured["env"]["PORT"] == "51200"
    assert captured["chat_kwargs"]["base_url"] == "http://127.0.0.1:51200"
    assert captured["chat_kwargs"]["model"] == "gemini-3.5-flash-low"
    assert captured["accounts_config"]["accounts"][0]["source"] == "oauth"
    assert captured["accounts_config"]["accounts"][0]["refreshToken"] == "rotated-refresh|user-project"
    assert "access-token-should-not-be-written" not in json.dumps(captured["accounts_config"])
    assert captured["terminated"] is True
    assert any(event["type"] == "terminal_log" and "fast Ask generation" in event["content"]["line"] for event in events)


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


def test_deepagents_runtime_ask_uses_fast_direct_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_DEEPAGENTS_ENABLED", "true")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_KEEP_WORKSPACE", "false")
    get_settings.cache_clear()

    captured: dict[str, object] = {}

    class _FastModel:
        def invoke(self, messages):
            captured["messages"] = messages
            return AIMessage(content="Fast answer from retrieved context.")

    runtime = DeepAgentsEditRuntime(model=_FastModel())
    events = list(
        runtime.stream(
            EditRuntimeRequest(
                prompt="What is MATONTO?",
                thread_id="thread-deepagents-ask",
                trace_id="trace-deepagents-ask",
                task="ask",
                retrieved_context="MATONTO covers materials ontology terminology.",
                citation_labels=("MATONTO v2",),
            )
        )
    )

    assert any(event["type"] == "terminal_log" and "fast Ask generation" in event["content"]["line"] for event in events)
    prompt_text = captured["messages"][0].content
    assert "Answer fast" in prompt_text
    assert "Do not use tools" in prompt_text
    assert "MATONTO covers materials ontology terminology." in prompt_text
    validation_event = next(event["content"] for event in events if event["type"] == "validation_report")
    assert validation_event.get("runtime", {}).get("status", "passed") == "passed"
