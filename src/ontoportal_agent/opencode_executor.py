from __future__ import annotations

import json
import os
import re
import selectors
import signal
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

from rdflib import Graph

from .config import AgentSettings, get_settings

_ONTOLOGY_ARTIFACT_SUFFIXES = {".ttl", ".rdf", ".owl", ".json", ".yaml", ".yml", ".md", ".txt"}
_RDF_FORMAT_CANDIDATES = {
    ".ttl": ("turtle",),
    ".rdf": ("xml", "turtle", "n3"),
    ".owl": ("xml", "turtle", "n3"),
}
_SECRET_PATTERNS = (
    (re.compile(r"(?i)(api[_-]?key=)[^&\s\"']+"), r"\1[redacted]"),
    (re.compile(r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\"'\s,}]+"), r"\1[redacted]"),
    (re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s\"']+"), r"\1[redacted]"),
)
_USER_PROVIDER_ID = "matportal-user"
_USER_PROVIDER_API_KEY_ENV = "MATPORTAL_OPENCODE_API_KEY"
_MCP_API_KEY_ENV = "MATPORTAL_MCP_API_KEY"
_MCP_API_KEY_PLACEHOLDER = f"{{env:{_MCP_API_KEY_ENV}}}"
_OPENAI_COMPATIBLE_NPM = "@ai-sdk/openai-compatible"
_ONTOLOGY_TOOLKIT_DIR = "matportal-ontology-toolkit"
_OPENCODE_ENV_PASSTHROUGH = {
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NO_PROXY",
    "NODE_EXTRA_CA_CERTS",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TMP",
    "TMPDIR",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
}


@dataclass(frozen=True)
class OpenCodeProviderAuth:
    provider_id: str
    model: str
    api_key: str = field(repr=False)
    base_url: str
    name: str = "MatPortal user generation provider"
    env_api_key_name: str = _USER_PROVIDER_API_KEY_ENV

    @property
    def model_ref(self) -> str:
        return f"{self.provider_id}/{self.model}"


@dataclass
class OpenCodeExecutionResult:
    ok: bool
    workspace: str
    run_id: str
    expires_at: str
    session_id: str | None = None
    model: str | None = None
    final_text: str = ""
    exit_code: int = 0
    timed_out: bool = False
    console_lines: list[str] = field(default_factory=list)
    changed_files: list[dict[str, Any]] = field(default_factory=list)
    diff_summary: dict[str, Any] = field(default_factory=dict)
    artifact_candidates: list[dict[str, Any]] = field(default_factory=list)
    validation_report: dict[str, Any] = field(default_factory=dict)

    def execution_payload(self) -> dict[str, Any]:
        return {
            "mode": "opencode",
            "ok": self.ok,
            "run_id": self.run_id,
            "workspace": self.workspace,
            "session_id": self.session_id,
            "model": self.model,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "logs": self.console_lines,
            "changed_files": self.changed_files,
            "diff_summary": self.diff_summary,
            "artifact_candidates": self.artifact_candidates,
            "validation_report": self.validation_report,
            "expires_at": self.expires_at,
        }


class OpenCodeExecutor:
    def __init__(self, settings: AgentSettings | None = None, provider_auth: OpenCodeProviderAuth | None = None):
        self.settings = settings or get_settings()
        self.provider_auth = provider_auth

    def stream(
        self,
        *,
        prompt: str,
        thread_id: str | None,
        trace_id: str | None = None,
        task: str = "edit",
        retrieved_context: str = "",
        citation_labels: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        run_id = uuid.uuid4().hex
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=max(1, int(self.settings.opencode_artifact_retention_days)))
        ).isoformat()
        workspace = self._prepare_workspace(thread_id=thread_id, run_id=run_id)
        result = OpenCodeExecutionResult(
            ok=False,
            workspace=str(workspace),
            run_id=run_id,
            expires_at=expires_at,
            model=self._opencode_model_ref(),
        )
        yield {
            "type": "workspace_mode",
            "content": {
                "mode": "execution",
                "run_id": run_id,
                "workspace": str(workspace),
                "expires_at": expires_at,
                "title": "OpenCode workspace",
            },
        }
        yield {
            "type": "opencode_phase",
            "content": {
                "label": "Preparing workspace",
                "run_id": run_id,
                "workspace": str(workspace),
            },
        }
        command = self._command(
            prompt=prompt,
            workspace=workspace,
            task=task,
            retrieved_context=retrieved_context,
            citation_labels=citation_labels,
        )
        self._append_console_line(result, f"$ {' '.join(command)}")
        yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}

        process = subprocess.Popen(
            command,
            cwd=str(workspace),
            env=self._opencode_environment(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            result.timed_out = yield from self._stream_process_output(process=process, result=result)
            if not result.timed_out:
                process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()

        result.exit_code = -9 if result.timed_out else int(process.returncode or 0)
        self._finalize_workspace(workspace=workspace, result=result)

        yield {
            "type": "changed_files",
            "content": result.changed_files,
        }
        yield {
            "type": "diff_summary",
            "content": result.diff_summary,
        }
        yield {
            "type": "artifact_candidates",
            "content": result.artifact_candidates,
        }
        yield {
            "type": "validation_report",
            "content": result.validation_report,
        }

        if result.exit_code == 0 and result.validation_report.get("ok", False):
            result.ok = True
            yield {
                "type": "opencode_phase",
                "content": {
                    "label": "Workspace complete",
                    "workspace": str(workspace),
                },
            }
        else:
            failure = (
                f"OpenCode exited with code {result.exit_code}."
                if result.exit_code != 0
                else "OpenCode artifact validation failed."
            )
            self._append_console_line(result, failure)
            yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
            yield {
                "type": "opencode_phase",
                "content": {
                    "label": "Workspace failed",
                    "workspace": str(workspace),
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                },
            }

        return result

    def _prepare_workspace(self, *, thread_id: str | None, run_id: str | None = None) -> Path:
        root = self.settings.ontology_workdir / self.settings.opencode_workspace_subdir
        root.mkdir(parents=True, exist_ok=True)
        self._chmod_private(root, 0o700)
        token = str(thread_id or "standalone").replace("/", "-")
        workspace = root / f"{token}-{run_id or int(time.time())}"
        workspace.mkdir(parents=True, exist_ok=True)
        self._chmod_private(workspace, 0o700)

        config = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {
                self.settings.opencode_mcp_name: self._opencode_mcp_config(),
            },
        }
        if self.provider_auth:
            config["provider"] = {
                self.provider_auth.provider_id: {
                    "npm": _OPENAI_COMPATIBLE_NPM,
                    "name": self.provider_auth.name,
                    "options": {
                        "baseURL": self.provider_auth.base_url,
                        "apiKey": f"{{env:{self.provider_auth.env_api_key_name}}}",
                    },
                    "models": {
                        self.provider_auth.model: {
                            "name": self.provider_auth.model,
                        },
                    },
                }
            }
        config_path = workspace / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._chmod_private(config_path, 0o600)
        (workspace / "README.md").write_text(
            "\n".join(
                [
                    "# OpenCode Ontology Workspace",
                    "",
                    "This workspace is disposable.",
                    "- Inspect ontology state through the ontoportal_api MCP server.",
                    f"- Use `{_ONTOLOGY_TOOLKIT_DIR}/` for proposal templates and review checklists.",
                    "- Write proposed ontology changes into files under this directory.",
                    "- Copy toolkit templates into new proposal files; do not edit toolkit files unless asked.",
                    "- Do not commit, push, or modify remotes.",
                ]
            ),
            encoding="utf-8",
        )
        self._write_ontology_toolkit(workspace)
        self._init_git_repo(workspace)
        self._write_workspace_excludes(workspace)
        return workspace

    def _chmod_private(self, path: Path, mode: int) -> None:
        try:
            os.chmod(path, mode)
        except OSError:
            pass

    def _write_workspace_excludes(self, workspace: Path) -> None:
        exclude_file = workspace / ".git" / "info" / "exclude"
        if not exclude_file.exists():
            return
        existing = exclude_file.read_text(encoding="utf-8")
        entries = [
            "",
            "# MatPortal assistant runtime state",
            ".opencode-home/",
            ".opencode-state/",
            ".opencode-cache/",
            ".opencode/",
            ".config/",
            ".cache/",
            ".local/",
        ]
        additions = [entry for entry in entries if entry and entry not in existing]
        if additions:
            exclude_file.write_text(f"{existing.rstrip()}\n" + "\n".join(additions) + "\n", encoding="utf-8")

    def _write_ontology_toolkit(self, workspace: Path) -> None:
        toolkit = workspace / _ONTOLOGY_TOOLKIT_DIR
        toolkit.mkdir(parents=True, exist_ok=True)
        self._chmod_private(toolkit, 0o700)
        files = {
            "README.md": self._ontology_toolkit_readme(),
            "proposal-template.ttl": self._ontology_turtle_template(),
            "operator-report-template.md": self._operator_report_template(),
            "review-checklist.json": self._review_checklist_template(),
        }
        for name, content in files.items():
            (toolkit / name).write_text(content, encoding="utf-8")

    def _ontology_toolkit_readme(self) -> str:
        return "\n".join(
            [
                "# MatPortal Ontology Edit Toolkit",
                "",
                "Use these files as references when preparing ontology edit artifacts.",
                "",
                "Expected workflow:",
                "1. Inspect the relevant ontology/API state through the configured MCP server.",
                "2. Copy `proposal-template.ttl` into a new `.ttl` file for RDF/Turtle proposals.",
                "3. Copy `operator-report-template.md` into a new review note when the edit needs explanation.",
                "4. Keep generated artifacts at the workspace root or in a purpose-named subdirectory.",
                "5. Finish with a short summary naming changed files, validation status, and assumptions.",
                "",
                "Do not put credentials, API keys, or absolute local paths into generated artifacts.",
                "Do not edit toolkit files unless the user explicitly asks for a toolkit change.",
                "",
            ]
        )

    def _ontology_turtle_template(self) -> str:
        return "\n".join(
            [
                "@prefix dcterms: <http://purl.org/dc/terms/> .",
                "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
                "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
                "@prefix matportal: <https://w3id.org/matportal/assistant/> .",
                "",
                "matportal:OntologyEditProposal",
                "    a owl:Ontology ;",
                '    dcterms:title "MatPortal ontology edit proposal" ;',
                '    dcterms:description "Replace this template with the requested ontology edit proposal." .',
                "",
                "# Add proposed classes, properties, axioms, mappings, or annotations below.",
                "# Prefer stable source IRIs from the inspected ontology when extending existing terms.",
                "",
            ]
        )

    def _operator_report_template(self) -> str:
        return "\n".join(
            [
                "# Operator Review Notes",
                "",
                "## Request",
                "- User request:",
                "",
                "## Inspected Context",
                "- Ontologies/API endpoints checked:",
                "- Relevant source terms:",
                "",
                "## Proposed Artifacts",
                "- Files changed:",
                "- Validation result:",
                "",
                "## Assumptions",
                "-",
                "",
                "## Follow-up",
                "-",
                "",
            ]
        )

    def _review_checklist_template(self) -> str:
        return json.dumps(
            {
                "checks": [
                    "Relevant ontology/API state inspected",
                    "Generated RDF parses successfully when applicable",
                    "New terms have labels and definitions where appropriate",
                    "Mappings or external references use stable IRIs",
                    "Operator notes list assumptions and follow-up actions",
                    "No secrets or absolute local paths are present",
                ]
            },
            indent=2,
        ) + "\n"

    def _opencode_environment(self, workspace: Path) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if key in _OPENCODE_ENV_PASSTHROUGH and value is not None
        }
        env.setdefault("PATH", os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))
        home = workspace / ".opencode-home"
        config_home = home / ".config"
        data_home = home / ".local" / "share"
        cache_home = home / ".cache"
        for directory in (home, config_home, data_home, cache_home):
            directory.mkdir(parents=True, exist_ok=True)
            self._chmod_private(directory, 0o700)
        env["HOME"] = str(home)
        env["XDG_CONFIG_HOME"] = str(config_home)
        env["XDG_DATA_HOME"] = str(data_home)
        env["XDG_CACHE_HOME"] = str(cache_home)
        env[_MCP_API_KEY_ENV] = self.settings.ontoportal_api_key
        if self.provider_auth:
            env[self.provider_auth.env_api_key_name] = self.provider_auth.api_key
        return env

    def _stream_process_output(
        self,
        *,
        process: subprocess.Popen[str],
        result: OpenCodeExecutionResult,
    ) -> Iterator[dict[str, Any]]:
        stdout = process.stdout
        if stdout is None:
            return False

        timeout_seconds = max(1, int(self.settings.opencode_run_timeout_seconds))
        deadline = time.monotonic() + timeout_seconds
        selector = selectors.DefaultSelector()
        timed_out = False
        selector.register(stdout, selectors.EVENT_READ)
        try:
            while True:
                if process.poll() is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                ready = selector.select(timeout=min(0.5, remaining))
                if not ready:
                    continue
                raw_line = stdout.readline()
                if not raw_line:
                    continue
                yield from self._events_for_stdout_line(raw_line=raw_line, result=result)

            if timed_out:
                self._append_console_line(
                    result,
                    f"OpenCode timed out after {timeout_seconds} seconds; terminating workspace run.",
                )
                yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
                self._terminate_process_group(process)

            for raw_line in stdout:
                yield from self._events_for_stdout_line(raw_line=raw_line, result=result)
        finally:
            try:
                selector.unregister(stdout)
            except Exception:
                pass
            selector.close()

        return timed_out

    def _events_for_stdout_line(
        self,
        *,
        raw_line: str,
        result: OpenCodeExecutionResult,
    ) -> Iterator[dict[str, Any]]:
        line = raw_line.strip()
        if not line:
            return
        emitted = self._handle_stdout_line(line=line, result=result)
        for event in emitted:
            yield event

    def _terminate_process_group(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (AttributeError, ProcessLookupError, OSError):
            try:
                process.terminate()
            except OSError:
                return
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, OSError):
            try:
                process.kill()
            except OSError:
                return
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def _opencode_mcp_config(self) -> dict[str, Any]:
        if self.settings.opencode_mcp_mode == "local":
            return {
                "type": "local",
                "command": self._opencode_mcp_local_command(),
                "environment": self._opencode_mcp_local_environment(),
                "enabled": True,
                "timeout": self.settings.opencode_mcp_timeout_ms,
            }

        return {
            "type": "remote",
            "url": self._opencode_mcp_url(),
            "enabled": True,
            "timeout": self.settings.opencode_mcp_timeout_ms,
        }

    def _opencode_mcp_url(self) -> str:
        query = "&".join(
            [
                f"api_key={_MCP_API_KEY_PLACEHOLDER}",
                urlencode({"base_url": self.settings.ontoportal_api_base.rstrip("/")}),
            ]
        )
        return f"{self.settings.opencode_mcp_url}?{query}"

    def _opencode_mcp_local_command(self) -> list[str]:
        root = self.settings.opencode_mcp_server_root
        if root is None:
            raise RuntimeError("ONTOAGENT_OPENCODE_MCP_SERVER_ROOT is required when OPENCODE_MCP_MODE=local")
        return [
            "sh",
            "-lc",
            f"cd {shlex.quote(str(root))} && exec {shlex.quote(self.settings.opencode_mcp_python)} mcp_server.py",
        ]

    def _opencode_mcp_local_environment(self) -> dict[str, str]:
        return {
            "PYTHONUNBUFFERED": "1",
            "MCP_TRANSPORT": self.settings.opencode_mcp_transport,
            "ONTO_PORTAL_BASE_URL": self.settings.ontoportal_api_base.rstrip("/"),
            "ONTO_PORTAL_API_KEY": _MCP_API_KEY_PLACEHOLDER,
        }

    def _command(
        self,
        *,
        prompt: str,
        workspace: Path,
        task: str = "edit",
        retrieved_context: str = "",
        citation_labels: list[str] | None = None,
    ) -> list[str]:
        command = [
            self.settings.opencode_path,
            "run",
            "--format",
            "json",
            "--dir",
            str(workspace),
        ]
        model_ref = self._opencode_model_ref()
        if model_ref:
            command.extend(["--model", model_ref])
        command.append(
            self._opencode_prompt(
                prompt,
                task=task,
                retrieved_context=retrieved_context,
                citation_labels=citation_labels,
            )
        )
        return command

    def _opencode_model_ref(self) -> str:
        if self.provider_auth:
            return self.provider_auth.model_ref
        return self.settings.opencode_model or ""

    def _opencode_prompt(
        self,
        prompt: str,
        *,
        task: str = "edit",
        retrieved_context: str = "",
        citation_labels: list[str] | None = None,
    ) -> str:
        if str(task or "").strip().lower() == "ask":
            citations = "\n".join(f"- {item}" for item in (citation_labels or []) if str(item or "").strip())
            return (
                "You are answering a MatPortal assistant Ask request.\n"
                "Mandatory rules:\n"
                "- Use only the retrieved context provided below.\n"
                "- Do not inspect external resources or call tools.\n"
                "- Do not write, edit, delete, or create files.\n"
                "- If the retrieved context is weak or missing, say what is missing.\n"
                "- Keep the answer concise and operator-facing.\n\n"
                f"User question:\n{prompt.strip()}\n\n"
                f"Retrieved context:\n{str(retrieved_context or '').strip() or '(none)'}\n\n"
                f"Citations:\n{citations or '- none'}\n\n"
                "Return only the answer text."
            )

        mcp_name = self.settings.opencode_mcp_name
        return (
            "You are preparing an ontology-edit proposal for MatPortal.\n"
            "Mandatory rules:\n"
            f"- Use the {mcp_name} MCP server to inspect the ontology/API state relevant to this request before editing.\n"
            "- Work only inside the current workspace.\n"
            f"- Use the `{_ONTOLOGY_TOOLKIT_DIR}/` templates and checklist as references for ontology artifacts and review notes.\n"
            "- Copy toolkit templates into new proposal files; do not edit toolkit files unless the user explicitly asks.\n"
            "- Do not commit, push, or access git remotes.\n"
            "- Write proposed artifacts and notes into files in this workspace.\n"
            "- Prefer Turtle (.ttl) unless the user explicitly requests another format.\n"
            "- Finish with a concise operator-facing summary of the proposed changes.\n\n"
            f"User request:\n{prompt.strip()}"
        )

    def _init_git_repo(self, workspace: Path) -> None:
        self._run_git(workspace, "init")
        self._run_git(workspace, "config", "user.name", "MatPortal Assistant")
        self._run_git(workspace, "config", "user.email", "assistant@matportal.invalid")
        self._run_git(workspace, "add", "-A")
        self._run_git(workspace, "commit", "--allow-empty", "-m", "Workspace baseline")

    def _run_git(self, workspace: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    def _handle_stdout_line(self, *, line: str, result: OpenCodeExecutionResult) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._append_console_line(result, line)
            return [{"type": "terminal_log", "content": {"line": result.console_lines[-1]}}]

        event_type = str(payload.get("type") or "").strip()
        part = payload.get("part") or {}
        if payload.get("sessionID") and not result.session_id:
            result.session_id = str(payload.get("sessionID"))

        if event_type == "step_start":
            label = "OpenCode is planning the workspace run"
            self._append_console_line(result, label)
            events.append({"type": "opencode_phase", "content": {"label": label}})
            events.append({"type": "terminal_log", "content": {"line": label}})
            return events

        if event_type == "tool_use":
            tool = str(part.get("tool") or "tool").strip()
            state = part.get("state") or {}
            title = str(state.get("title") or state.get("input", {}).get("description") or tool).strip()
            header = f"[{tool}] {title}"
            self._append_console_line(result, header)
            events.append({"type": "opencode_phase", "content": {"label": f"Running {tool}"}})
            events.append({"type": "terminal_log", "content": {"line": header}})
            for detail_line in self._tool_detail_lines(tool=tool, state=state):
                self._append_console_line(result, detail_line)
                events.append({"type": "terminal_log", "content": {"line": result.console_lines[-1]}})
            return events

        if event_type == "text":
            text = str(part.get("text") or "").strip()
            if text:
                result.final_text = text
            return events

        if event_type == "step_finish":
            reason = str(part.get("reason") or "stop").strip()
            label = f"OpenCode step finished ({reason})"
            self._append_console_line(result, label)
            events.append({"type": "opencode_phase", "content": {"label": label}})
            events.append({"type": "terminal_log", "content": {"line": label}})
            tokens = (part.get("tokens") or {})
            reasoning = tokens.get("reasoning")
            total = tokens.get("total")
            if total is not None or reasoning is not None:
                detail = f"tokens total={total or 0} reasoning={reasoning or 0}"
                self._append_console_line(result, detail)
                events.append({"type": "terminal_log", "content": {"line": detail}})
            return events

        self._append_console_line(result, line)
        events.append({"type": "terminal_log", "content": {"line": result.console_lines[-1]}})
        return events

    def _tool_detail_lines(self, *, tool: str, state: dict[str, Any]) -> list[str]:
        details: list[str] = []
        input_payload = state.get("input") or {}
        read_file_path = ""
        if tool == "bash":
            command = str(input_payload.get("command") or "").strip()
            if command:
                details.append(f"command: {command}")
        elif tool == "read":
            read_file_path = str(input_payload.get("filePath") or "").strip()
            if read_file_path:
                details.append(f"file: {read_file_path}")

        output_text = str(state.get("output") or "").strip()
        if output_text:
            if tool == "read" and "/root/.local/share/opencode/tool-output/" in read_file_path:
                details.append("  opened saved tool-output file for follow-up inspection")
                return details
            details.extend(self._summarize_tool_output(tool=tool, output_text=output_text))

        metadata = state.get("metadata") or {}
        exit_code = metadata.get("exit")
        if exit_code is not None:
            details.append(f"exit: {exit_code}")
        return details

    def _summarize_tool_output(self, *, tool: str, output_text: str) -> list[str]:
        if tool.startswith("ontoportal_api_"):
            parsed = self._safe_json(output_text)
            summarized = self._summarize_ontoportal_output(parsed)
            if summarized:
                return [f"  {line}" for line in summarized]

        output_lines = [line for line in output_text.splitlines() if line.strip()]
        if not output_lines:
            return []

        shown = output_lines[:4]
        details = [f"  {self._truncate_console_line(line)}" for line in shown]
        if len(output_lines) > len(shown):
            details.append(f"  ... {len(output_lines) - len(shown)} more line(s)")
        return details

    def _summarize_ontoportal_output(self, payload: Any) -> list[str]:
        if isinstance(payload, dict):
            lines: list[str] = []
            acronym = str(payload.get("acronym") or "").strip()
            name = str(payload.get("name") or payload.get("prefLabel") or "").strip()
            if acronym or name:
                joined = " - ".join(part for part in [acronym, name] if part)
                if joined:
                    lines.append(self._truncate_console_line(joined))

            if "totalCount" in payload or "pageCount" in payload or "page" in payload:
                page = payload.get("page")
                page_count = payload.get("pageCount")
                total = payload.get("totalCount")
                counts = []
                if page is not None and page_count is not None:
                    counts.append(f"page {page}/{page_count}")
                if total is not None:
                    counts.append(f"{total} total items")
                if counts:
                    lines.append(", ".join(counts))

            collection = payload.get("collection")
            if isinstance(collection, list):
                lines.append(f"collection size {len(collection)}")
                first = collection[0] if collection else None
                if isinstance(first, dict):
                    first_label = str(first.get("prefLabel") or first.get("label") or first.get("name") or "").strip()
                    first_id = str(first.get("@id") or first.get("id") or "").strip()
                    first_summary = " | ".join(part for part in [first_label, first_id] if part)
                    if first_summary:
                        lines.append(f"first item {self._truncate_console_line(first_summary)}")

            if lines:
                return lines[:4]

        if isinstance(payload, list):
            return [f"list size {len(payload)}"]

        return []

    def _safe_json(self, raw: str) -> Any | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _truncate_console_line(self, value: str, max_chars: int = 180) -> str:
        clean = " ".join(str(value).split())
        if len(clean) <= max_chars:
            return clean
        return f"{clean[: max_chars - 3]}..."

    def _append_console_line(self, result: OpenCodeExecutionResult, line: str) -> None:
        clean = self._redact_sensitive(str(line or "").rstrip())
        if not clean:
            return
        result.console_lines.append(clean)
        if len(result.console_lines) > self.settings.opencode_max_log_lines:
            del result.console_lines[:-self.settings.opencode_max_log_lines]

    def _redact_sensitive(self, value: str) -> str:
        redacted = str(value or "")
        secrets = [self.settings.ontoportal_api_key, self.settings.openai_api_key]
        if self.provider_auth:
            secrets.append(self.provider_auth.api_key)
        for secret in secrets:
            secret_text = str(secret or "").strip()
            if secret_text:
                redacted = redacted.replace(secret_text, "[redacted]")
        for pattern, replacement in _SECRET_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    def _finalize_workspace(self, *, workspace: Path, result: OpenCodeExecutionResult) -> None:
        subprocess.run(["git", "add", "-A"], cwd=str(workspace), check=False, capture_output=True, text=True)
        status_output = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        result.changed_files = self._parse_changed_files(status_output, workspace=workspace)

        diff_stat = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        diff_text = subprocess.run(
            ["git", "diff", "--cached", "--no-color", "--unified=3"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        result.diff_summary = {
            "stat": diff_stat,
            "preview": diff_text[: self.settings.opencode_max_diff_chars].strip(),
            "truncated": len(diff_text) > self.settings.opencode_max_diff_chars,
        }
        result.artifact_candidates = self._artifact_candidates(result.changed_files)
        result.validation_report = self._build_validation_report(workspace=workspace, changed_files=result.changed_files)

        if not self.settings.opencode_keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)

    def _build_validation_report(self, *, workspace: Path, changed_files: list[dict[str, Any]]) -> dict[str, Any]:
        checked_files: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        for item in changed_files:
            path_text = str(item.get("path") or "").strip()
            if not path_text:
                continue
            suffix = Path(path_text).suffix.lower()
            if suffix not in _ONTOLOGY_ARTIFACT_SUFFIXES:
                continue
            if str(item.get("status") or "").upper() == "D":
                checked_files.append({"path": path_text, "kind": suffix.lstrip("."), "status": "skipped", "message": "Deleted file"})
                continue

            path = self._workspace_file_path(workspace, path_text)
            if path is None:
                error = {"path": path_text, "message": "Path is outside the OpenCode workspace."}
                errors.append(error)
                checked_files.append({"path": path_text, "kind": suffix.lstrip("."), "status": "failed", "message": error["message"]})
                continue
            if not path.exists():
                error = {"path": path_text, "message": "Changed file is missing from the workspace."}
                errors.append(error)
                checked_files.append({"path": path_text, "kind": suffix.lstrip("."), "status": "failed", "message": error["message"]})
                continue

            if suffix in _RDF_FORMAT_CANDIDATES:
                entry = self._validate_rdf_file(path=path, display_path=path_text, formats=_RDF_FORMAT_CANDIDATES[suffix])
            elif suffix == ".json":
                entry = self._validate_json_file(path=path, display_path=path_text)
            elif suffix in {".yaml", ".yml"}:
                entry = self._validate_yaml_file(path=path, display_path=path_text)
            else:
                entry = {
                    "path": path_text,
                    "kind": suffix.lstrip("."),
                    "status": "skipped",
                    "message": "Plain text artifact; no syntax validator available.",
                }

            checked_files.append(entry)
            if entry.get("status") == "failed":
                errors.append({"path": path_text, "message": str(entry.get("message") or "Validation failed.")})
            elif entry.get("status") == "skipped":
                warnings.append({"path": path_text, "message": str(entry.get("message") or "Validation skipped.")})

        passed = sum(1 for item in checked_files if item.get("status") == "passed")
        failed = sum(1 for item in checked_files if item.get("status") == "failed")
        skipped = sum(1 for item in checked_files if item.get("status") == "skipped")
        if failed:
            status_text = "failed"
        elif passed:
            status_text = "passed"
        else:
            status_text = "skipped"
        return {
            "ok": failed == 0,
            "status": status_text,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checked_files": checked_files,
            "errors": errors,
            "warnings": warnings,
            "summary": f"{passed} passed, {failed} failed, {skipped} skipped",
        }

    def _workspace_file_path(self, workspace: Path, path_text: str) -> Path | None:
        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        root = workspace.resolve()
        candidate = (workspace / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def _validate_rdf_file(self, *, path: Path, display_path: str, formats: tuple[str, ...]) -> dict[str, Any]:
        last_error = ""
        for rdf_format in formats:
            graph = Graph()
            try:
                graph.parse(path, format=rdf_format)
                return {
                    "path": display_path,
                    "kind": path.suffix.lstrip("."),
                    "status": "passed",
                    "parser": rdf_format,
                    "triples": len(graph),
                }
            except Exception as exc:
                last_error = self._format_validation_error(exc, workspace=path.parent)
        return {
            "path": display_path,
            "kind": path.suffix.lstrip("."),
            "status": "failed",
            "message": last_error or "RDF parser rejected the artifact.",
        }

    def _validate_json_file(self, *, path: Path, display_path: str) -> dict[str, Any]:
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return {"path": display_path, "kind": "json", "status": "passed", "parser": "json"}
        except Exception as exc:
            return {
                "path": display_path,
                "kind": "json",
                "status": "failed",
                "message": self._format_validation_error(exc, workspace=path.parent),
            }

    def _validate_yaml_file(self, *, path: Path, display_path: str) -> dict[str, Any]:
        try:
            import yaml  # type: ignore
        except Exception:
            return {
                "path": display_path,
                "kind": path.suffix.lstrip("."),
                "status": "skipped",
                "message": "PyYAML is not installed in this runtime.",
            }
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            return {"path": display_path, "kind": path.suffix.lstrip("."), "status": "passed", "parser": "yaml"}
        except Exception as exc:
            return {
                "path": display_path,
                "kind": path.suffix.lstrip("."),
                "status": "failed",
                "message": self._format_validation_error(exc, workspace=path.parent),
            }

    def _format_validation_error(self, exc: Exception, *, workspace: Path) -> str:
        message = " ".join(str(exc).split())
        if message:
            message = message.replace(str(workspace), "<workspace>")
        return self._truncate_console_line(message or exc.__class__.__name__, max_chars=240)

    def _parse_changed_files(self, raw: str, *, workspace: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            status = line[:2].strip() or "M"
            path_text = line[3:].strip()
            if " -> " in path_text:
                path_text = path_text.rsplit(" -> ", 1)[-1].strip()
            path = Path(path_text)
            items.append(
                {
                    "status": status,
                    "path": path_text,
                    "kind": path.suffix.lstrip(".") or "file",
                }
            )
        return items

    def _artifact_candidates(self, changed_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for item in changed_files:
            suffix = Path(str(item.get("path") or "")).suffix.lower()
            if suffix not in _ONTOLOGY_ARTIFACT_SUFFIXES:
                continue
            candidates.append(
                {
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "status": item.get("status"),
                }
            )
        return candidates
