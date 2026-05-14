import importlib
import json
import stat
from pathlib import Path

import pytest
from rdflib import Graph

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.config import get_settings
from ontoportal_agent.opencode_executor import OpenCodeAccountAuth, OpenCodeExecutionResult, OpenCodeExecutor, OpenCodeProviderAuth


def test_prepare_workspace_writes_opencode_mcp_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_OPENCODE_MCP_URL", "https://mcp.matportal.org/mcp")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_BASE", "https://data.dev.matportal.org/")
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    workspace = executor._prepare_workspace(thread_id="thread-42")

    config = json.loads((workspace / "opencode.json").read_text(encoding="utf-8"))
    server = config["mcp"]["ontoportal_api"]
    assert server["type"] == "remote"
    assert server["enabled"] is True
    assert "api_key=test-ontoportal-key" in server["url"]
    assert "base_url=https%3A%2F%2Fdata.dev.matportal.org" in server["url"]
    assert (workspace / ".git").exists()
    assert (workspace / "README.md").exists()
    assert stat.S_IMODE(workspace.stat().st_mode) == 0o700
    assert stat.S_IMODE((workspace / "opencode.json").stat().st_mode) == 0o600
    toolkit = workspace / "matportal-ontology-toolkit"
    assert (toolkit / "README.md").exists()
    assert (toolkit / "proposal-template.ttl").exists()
    assert (toolkit / "operator-report-template.md").exists()
    assert (toolkit / "review-checklist.json").exists()
    assert stat.S_IMODE(toolkit.stat().st_mode) == 0o700
    Graph().parse(toolkit / "proposal-template.ttl", format="turtle")
    checklist = json.loads((toolkit / "review-checklist.json").read_text(encoding="utf-8"))
    assert "No secrets or absolute local paths are present" in checklist["checks"]
    excludes = (workspace / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert ".opencode-home/" in excludes
    assert ".opencode-state/" in excludes


def test_prepare_workspace_writes_local_opencode_mcp_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_BASE", "https://data.dev.matportal.org/")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_OPENCODE_MCP_MODE", "local")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_MCP_SERVER_ROOT", "/opt/ontoportal-api-mcp")
    monkeypatch.setenv("ONTOAGENT_OPENCODE_MCP_PYTHON", "/venv/bin/python")
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    workspace = executor._prepare_workspace(thread_id="thread-local")

    config = json.loads((workspace / "opencode.json").read_text(encoding="utf-8"))
    server = config["mcp"]["ontoportal_api"]
    assert server["type"] == "local"
    assert server["command"][0:2] == ["sh", "-lc"]
    assert "cd /opt/ontoportal-api-mcp" in server["command"][2]
    assert "/venv/bin/python mcp_server.py" in server["command"][2]
    assert server["environment"]["MCP_TRANSPORT"] == "stdio"
    assert server["environment"]["ONTO_PORTAL_BASE_URL"] == "https://data.dev.matportal.org"
    assert server["environment"]["ONTO_PORTAL_API_KEY"] == "test-ontoportal-key"


def test_prepare_workspace_writes_user_provider_without_literal_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "deployment-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    auth = OpenCodeProviderAuth(
        provider_id="matportal-user",
        model="gemini-3.1-pro-preview",
        api_key="user-generation-secret",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    executor = OpenCodeExecutor(provider_auth=auth)
    workspace = executor._prepare_workspace(thread_id="thread-user-provider")

    raw_config = (workspace / "opencode.json").read_text(encoding="utf-8")
    config = json.loads(raw_config)
    provider = config["provider"]["matportal-user"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert provider["options"]["apiKey"] == "{env:MATPORTAL_OPENCODE_API_KEY}"
    assert "gemini-3.1-pro-preview" in provider["models"]
    assert "user-generation-secret" not in raw_config

    env = executor._opencode_environment(workspace)
    assert env["MATPORTAL_OPENCODE_API_KEY"] == "user-generation-secret"

    command = executor._command(prompt="Draft an edit", workspace=workspace)
    assert command[command.index("--model") + 1] == "matportal-user/gemini-3.1-pro-preview"


def test_opencode_ask_prompt_uses_backend_retrieved_context(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    prompt = executor._opencode_prompt(
        "Which ontology should I use for aluminium?",
        task="ask",
        retrieved_context="Use MATONTO for materials terms.",
        citation_labels=["MATONTO v2.0"],
    )

    assert "Use only the retrieved context" in prompt
    assert "Do not write, edit, delete, or create files" in prompt
    assert "Use MATONTO for materials terms." in prompt
    assert "- MATONTO v2.0" in prompt
    assert "Use the ontoportal_api MCP server" not in prompt
    assert "matportal-ontology-toolkit" not in prompt


def test_opencode_edit_prompt_references_ontology_toolkit(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    prompt = executor._opencode_prompt("Draft a Turtle proposal for aluminium alloys.", task="edit")

    assert "Use the ontoportal_api MCP server" in prompt
    assert "matportal-ontology-toolkit/" in prompt
    assert "Copy toolkit templates into new proposal files" in prompt
    assert "User request:\nDraft a Turtle proposal for aluminium alloys." in prompt


def test_ontoportal_tool_output_is_summarized(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    details = executor._tool_detail_lines(
        tool="ontoportal_api_getOntologyClasses",
        state={
            "output": json.dumps(
                {
                    "page": 1,
                    "pageCount": 20,
                    "totalCount": 980,
                    "collection": [
                        {
                            "prefLabel": "portion of beryllium",
                            "@id": "https://w3id.org/pmd/co/PMD_0020080",
                        }
                    ],
                }
            )
        },
    )

    assert "  page 1/20, 980 total items" in details
    assert "  collection size 1" in details
    assert any("portion of beryllium" in line for line in details)
    assert all(len(line) < 220 for line in details)


def test_reading_saved_tool_output_is_collapsed(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    details = executor._tool_detail_lines(
        tool="read",
        state={
            "input": {"filePath": "/root/.local/share/opencode/tool-output/tool_abc123"},
            "output": "<path>/root/.local/share/opencode/tool-output/tool_abc123</path><content>huge payload</content>",
        },
    )

    assert details == [
        "file: /root/.local/share/opencode/tool-output/tool_abc123",
        "  opened saved tool-output file for follow-up inspection",
    ]


def test_console_logs_redact_configured_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "ontoportal-secret")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    result = OpenCodeExecutionResult(
        ok=False,
        workspace=str(tmp_path),
        run_id="run-redaction",
        expires_at="2999-01-01T00:00:00+00:00",
    )

    executor._append_console_line(
        result,
        "api_key=ontoportal-secret Authorization: Bearer openai-secret literal=ontoportal-secret",
    )

    assert result.console_lines == [
        "api_key=[redacted] Authorization: Bearer [redacted] literal=[redacted]"
    ]


def test_console_logs_redact_user_provider_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "deployment-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "ontoportal-secret")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor(
        provider_auth=OpenCodeProviderAuth(
            provider_id="matportal-user",
            model="gpt-5.2",
            api_key="user-openai-secret",
            base_url="https://api.openai.com/v1",
        )
    )
    result = OpenCodeExecutionResult(
        ok=False,
        workspace=str(tmp_path),
        run_id="run-user-redaction",
        expires_at="2999-01-01T00:00:00+00:00",
    )

    executor._append_console_line(result, "Authorization: Bearer user-openai-secret literal=user-openai-secret")

    assert result.console_lines == ["Authorization: Bearer [redacted] literal=[redacted]"]


def test_execution_payload_does_not_include_provider_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "deployment-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "ontoportal-secret")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor(
        provider_auth=OpenCodeProviderAuth(
            provider_id="matportal-user",
            model="gemini-2.5-pro",
            api_key="user-gemini-secret",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
    )
    workspace = executor._prepare_workspace(thread_id="secret-payload")
    raw_config = (workspace / "opencode.json").read_text(encoding="utf-8")
    result = OpenCodeExecutionResult(
        ok=True,
        workspace=str(workspace),
        run_id="run-secret-payload",
        expires_at="2999-01-01T00:00:00+00:00",
    )
    executor._append_console_line(result, "literal user-gemini-secret should be redacted")

    payload_text = json.dumps(result.execution_payload())

    assert "user-gemini-secret" not in raw_config
    assert "{env:MATPORTAL_OPENCODE_API_KEY}" in raw_config
    assert "user-gemini-secret" not in payload_text


def test_opencode_account_auth_writes_isolated_auth_files(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "deployment-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "ontoportal-secret")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor(
        account_auth=OpenCodeAccountAuth(
            kind="gemini_antigravity",
            opencode_auth_json='{"provider":"antigravity","token":"antigravity-token"}',
            codex_auth_json='{"tokens":{"access_token":"codex-token"}}',
        )
    )
    workspace = executor._prepare_workspace(thread_id="account-auth")
    env = executor._opencode_environment(workspace)

    opencode_auth = workspace / ".opencode-home" / ".local" / "share" / "opencode" / "auth.json"
    codex_auth = workspace / ".codex-home" / "auth.json"
    assert json.loads(opencode_auth.read_text(encoding="utf-8"))["token"] == "antigravity-token"
    assert json.loads(codex_auth.read_text(encoding="utf-8"))["tokens"]["access_token"] == "codex-token"
    assert env["CODEX_HOME"] == str(workspace / ".codex-home")
    assert stat.S_IMODE(opencode_auth.stat().st_mode) == 0o600
    assert stat.S_IMODE(codex_auth.stat().st_mode) == 0o600


def test_opencode_environment_is_scoped_to_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_INTERNAL_API_TOKEN", "internal-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-secret")
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    workspace = executor._prepare_workspace(thread_id="thread-env")
    env = executor._opencode_environment(workspace)

    assert env["HOME"] == str(workspace / ".opencode-home")
    assert env["XDG_CONFIG_HOME"] == str(workspace / ".opencode-home" / ".config")
    assert env["XDG_DATA_HOME"] == str(workspace / ".opencode-home" / ".local" / "share")
    assert env["XDG_CACHE_HOME"] == str(workspace / ".opencode-home" / ".cache")
    assert (workspace / ".opencode-home" / ".config").is_dir()
    assert "PATH" in env
    assert "ONTOAGENT_OPENAI_API_KEY" not in env
    assert "ONTOAGENT_ONTOPORTAL_API_KEY" not in env
    assert "ONTOAGENT_INTERNAL_API_TOKEN" not in env
    assert "DATABASE_URL" not in env
    assert "OPENAI_API_KEY" not in env


def test_opencode_stream_times_out_and_terminates_process_group(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ONTOAGENT_OPENCODE_RUN_TIMEOUT_SECONDS", "1")
    fake_opencode = tmp_path / "fake-opencode"
    fake_opencode.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json",
                "import time",
                "print(json.dumps({'type': 'step_start'}), flush=True)",
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )
    fake_opencode.chmod(0o755)
    monkeypatch.setenv("ONTOAGENT_OPENCODE_PATH", str(fake_opencode))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    stream = executor.stream(prompt="sleep", thread_id="thread-timeout")
    events = []
    try:
        while True:
            events.append(next(stream))
    except StopIteration as stop:
        result = stop.value

    assert result.timed_out is True
    assert result.exit_code == -9
    assert any("timed out after 1 seconds" in str(event) for event in events)
    assert any((event.get("content") or {}).get("timed_out") is True for event in events)


def test_validation_report_checks_rdf_json_and_text_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    workspace = executor._prepare_workspace(thread_id="thread-validation")
    (workspace / "proposal.ttl").write_text(
        "@prefix ex: <https://example.org/> .\nex:Thing a ex:Class .\n",
        encoding="utf-8",
    )
    (workspace / "broken.json").write_text('{"bad":', encoding="utf-8")
    (workspace / "notes.md").write_text("Operator notes", encoding="utf-8")

    report = executor._build_validation_report(
        workspace=workspace,
        changed_files=[
            {"status": "A", "path": "proposal.ttl", "kind": "ttl"},
            {"status": "A", "path": "broken.json", "kind": "json"},
            {"status": "A", "path": "notes.md", "kind": "md"},
        ],
    )

    by_path = {item["path"]: item for item in report["checked_files"]}
    assert report["ok"] is False
    assert report["status"] == "failed"
    assert by_path["proposal.ttl"]["status"] == "passed"
    assert by_path["proposal.ttl"]["parser"] == "turtle"
    assert by_path["proposal.ttl"]["triples"] == 1
    assert by_path["broken.json"]["status"] == "failed"
    assert by_path["notes.md"]["status"] == "skipped"
    assert str(workspace) not in json.dumps(report)


def test_validation_report_rejects_symlink_escape(monkeypatch, tmp_path):
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    monkeypatch.setenv("ONTOAGENT_ONTOLOGY_WORKDIR", str(tmp_path))
    get_settings.cache_clear()

    executor = OpenCodeExecutor()
    workspace = executor._prepare_workspace(thread_id="thread-validation-symlink")
    outside = tmp_path / "outside.ttl"
    outside.write_text("@prefix ex: <https://example.org/> .\n", encoding="utf-8")
    (workspace / "escape.ttl").symlink_to(outside)

    report = executor._build_validation_report(
        workspace=workspace,
        changed_files=[{"status": "A", "path": "escape.ttl", "kind": "ttl"}],
    )

    assert report["ok"] is False
    assert report["checked_files"][0]["status"] == "failed"
    assert report["checked_files"][0]["message"] == "Path is outside the OpenCode workspace."
