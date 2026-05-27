from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterator

from ..agent.options import AgentRuntimeOptions
from ..artifact_store import ArtifactAccessError, assert_artifact_safe_for_exposure, sanitize_artifact_path
from ..config import AgentSettings, get_settings
from ..opencode_executor import OpenCodeExecutionResult, OpenCodeExecutor
from .base import EditRuntimeCapabilities, EditRuntimeRequest


class PiRuntimeError(RuntimeError):
    status_code = 503


class PiEditRuntime:
    """pi.dev CLI based edit runtime with backend-owned artifact writes.

    Pi is intentionally run without built-in tools. The model returns a structured
    artifact bundle, and this adapter writes artifacts through MatPortal's existing
    workspace, path, secret-scan, diff, and validation gates.
    """

    capabilities = EditRuntimeCapabilities(
        runtime="pi",
        supports_sessions=False,
        supports_cancel=False,
        supports_mcp=False,
        supports_artifacts=True,
    )

    runtime = "pi"

    def __init__(
        self,
        *,
        settings: AgentSettings | None = None,
        runtime_options: AgentRuntimeOptions | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.runtime_options = runtime_options
        self.model = str(model or self.settings.pi_model or "").strip() or None
        self._workspace_manager = OpenCodeExecutor(settings=self.settings, mcp_servers=[])

    def stream(self, request: EditRuntimeRequest) -> Iterator[dict[str, Any]]:
        if not self.settings.pi_adapter_enabled:
            raise PiRuntimeError("Pi edit runtime is disabled by ONTOAGENT_PI_ADAPTER_ENABLED.")
        run_id = uuid.uuid4().hex
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=max(1, int(self.settings.opencode_artifact_retention_days)))
        ).isoformat()
        workspace = self._workspace_manager._prepare_workspace(
            thread_id=request.thread_id,
            run_id=run_id,
            resume_workspace=request.resume_workspace,
        )
        result = OpenCodeExecutionResult(
            ok=False,
            workspace=str(workspace),
            run_id=run_id,
            expires_at=expires_at,
            session_id=request.resume_session_id or run_id,
            model=self._model_label(),
            runtime=self.runtime,
        )

        yield {
            "type": "workspace_mode",
            "content": {
                "mode": "execution",
                "runtime": self.runtime,
                "run_id": run_id,
                "workspace": str(workspace),
                "expires_at": expires_at,
                "title": "Pi workspace",
            },
        }
        yield {"type": "opencode_phase", "content": {"label": "Preparing Pi workspace", "run_id": run_id, "workspace": str(workspace), "runtime": self.runtime}}

        try:
            final_text = yield from self._run_pi(request=request, workspace=workspace, result=result)
            summary, artifacts = self._parse_artifact_bundle(final_text, require_artifacts=request.task != "ask")
            if artifacts:
                self._write_artifacts(workspace=workspace, artifacts=artifacts)
            result.final_text = summary or "Pi prepared the ontology edit workspace."
            result.exit_code = 0
        except Exception as exc:  # noqa: BLE001 - normalize experimental runtime failures.
            result.exit_code = int(getattr(exc, "status_code", 1) or 1)
            result.final_text = str(exc) or exc.__class__.__name__
            self._append_console_line(result, f"Pi failed: {result.final_text}")
            yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}

        self._workspace_manager._finalize_workspace(workspace=workspace, result=result)
        yield {"type": "changed_files", "content": result.changed_files}
        yield {"type": "diff_summary", "content": result.diff_summary}
        yield {"type": "artifact_candidates", "content": result.artifact_candidates}
        yield {"type": "validation_report", "content": result.validation_report}

        result.ok = result.exit_code == 0 and bool(result.validation_report.get("ok", True))
        if result.ok:
            yield {"type": "opencode_phase", "content": {"label": "Pi workspace complete", "workspace": str(workspace), "runtime": self.runtime}}
        else:
            yield {
                "type": "opencode_phase",
                "content": {
                    "label": "Pi workspace failed",
                    "workspace": str(workspace),
                    "runtime": self.runtime,
                    "exit_code": result.exit_code,
                },
            }
        return result

    def _run_pi(self, *, request: EditRuntimeRequest, workspace: Path, result: OpenCodeExecutionResult) -> Iterator[str]:
        pi_path = str(self.settings.pi_path or "pi").strip() or "pi"
        if shutil.which(pi_path) is None and not Path(pi_path).exists():
            raise PiRuntimeError(f"Pi executable not found: {pi_path}")
        command = [
            pi_path,
            "--mode",
            "json",
            "--print",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            "--no-session",
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append(self._user_prompt(request))
        env = self._pi_environment(workspace)
        timeout = max(1, int(self.settings.opencode_run_timeout_seconds))
        self._append_console_line(result, "Pi run started.")
        yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            raise PiRuntimeError(f"Pi run timed out after {timeout} seconds.") from exc
        final_text = self._extract_text_from_json_stream(completed.stdout)
        stderr = str(completed.stderr or "").strip()
        if completed.returncode != 0:
            detail = stderr or final_text or f"exit code {completed.returncode}"
            raise PiRuntimeError(f"Pi exited with {completed.returncode}: {detail[:1000]}")
        if stderr:
            self._append_console_line(result, self._redact_log_line(stderr[:1000]))
            yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        self._append_console_line(result, "Pi run finished.")
        yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        return final_text

    def _pi_environment(self, workspace: Path) -> dict[str, str]:
        allowed = {
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
        env = {key: value for key, value in os.environ.items() if key in allowed and value}
        env.setdefault("PATH", os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))
        env["PI_OFFLINE"] = "1"
        env["PI_TELEMETRY"] = "0"
        session_dir = (workspace / str(self.settings.pi_session_subdir or ".pi-sessions")).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        env["PI_CODING_AGENT_SESSION_DIR"] = str(session_dir)
        return env

    def _extract_text_from_json_stream(self, stdout: str) -> str:
        final_message_text = ""
        deltas: list[str] = []
        for line in str(stdout or "").splitlines():
            clean = line.strip()
            if not clean:
                continue
            try:
                event = json.loads(clean)
            except JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "message_update":
                update = event.get("assistantMessageEvent") or {}
                if update.get("type") == "text_delta":
                    deltas.append(str(update.get("delta") or ""))
            if event_type in {"message_end", "turn_end"}:
                message = event.get("message") or {}
                if message.get("role") == "assistant":
                    final_message_text = self._stringify_content(message.get("content"))
            if event_type == "agent_end":
                messages = event.get("messages") or []
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        final_message_text = self._stringify_content(message.get("content"))
                        break
        return (final_message_text or "".join(deltas)).strip()

    def _parse_artifact_bundle(self, text: str, *, require_artifacts: bool = True) -> tuple[str, list[dict[str, str]]]:
        payload = self._extract_json_object(text)
        summary = str(payload.get("summary") or payload.get("final_text") or "").strip()
        artifacts_raw = payload.get("artifacts") or []
        if not isinstance(artifacts_raw, list):
            raise PiRuntimeError("Pi artifact bundle must contain an artifacts array.")
        artifacts: list[dict[str, str]] = []
        for item in artifacts_raw:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            content = item.get("content")
            if not path or content is None:
                continue
            artifacts.append({"path": path, "content": str(content)})
        if require_artifacts and not artifacts:
            raise PiRuntimeError("Pi did not return any artifacts to write.")
        return summary, artifacts

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        clean = str(text or "").strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            clean = fence.group(1).strip()
        try:
            payload = json.loads(clean)
        except JSONDecodeError:
            decoder = json.JSONDecoder()
            payload = None
            for index, char in enumerate(clean):
                if char != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(clean[index:])
                except JSONDecodeError:
                    continue
                payload = candidate
                break
            if payload is None:
                raise PiRuntimeError("Pi response did not contain a valid JSON artifact bundle.")
        if not isinstance(payload, dict):
            raise PiRuntimeError("Pi response artifact bundle must be a JSON object.")
        return payload

    def _write_artifacts(self, *, workspace: Path, artifacts: list[dict[str, str]]) -> None:
        workspace_root = workspace.resolve()
        for artifact in artifacts:
            safe = sanitize_artifact_path(artifact["path"])
            target = (workspace_root / safe).resolve()
            try:
                target.relative_to(workspace_root)
            except ValueError as exc:
                raise ArtifactAccessError("Artifact path escapes the Pi workspace.") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(artifact.get("content") or ""), encoding="utf-8")
            try:
                assert_artifact_safe_for_exposure(target)
            except Exception:
                target.unlink(missing_ok=True)
                raise

    def _user_prompt(self, request: EditRuntimeRequest) -> str:
        context = str(request.retrieved_context or "").strip()
        labels = ", ".join(request.citation_labels)
        instructions = [
            "You are the MatPortal Pi edit runtime. You cannot use tools in this run.",
            "Prepare reviewable ontology assistant artifacts only; never publish, commit, push, or mutate live services.",
            "Return only one JSON object, with no prose outside JSON.",
            "JSON schema: {\"summary\": string, \"artifacts\": [{\"path\": string, \"content\": string}]}",
            "Required artifacts for edit tasks: edit-plan.json, evidence-ledger.json, operator-report.md, validation-summary.json, draft-submission.md, and at least one ontology artifact when a content change is requested.",
            "Artifacts must not contain secrets, API keys, OAuth tokens, local absolute paths, kubeconfigs, or runtime config.",
            "Use relative artifact paths only.",
        ]
        if context:
            instructions.extend(["Retrieved MatPortal context:", context[:120_000]])
        if labels:
            instructions.append(f"Citation labels available: {labels}")
        instructions.extend(["User request:", request.prompt])
        return "\n".join(instructions)

    def _model_label(self) -> str:
        return f"pi/{self.model or 'default'}"

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content or "")

    def _append_console_line(self, result: OpenCodeExecutionResult, line: str) -> None:
        result.console_lines.append(self._redact_log_line(line))
        max_lines = max(1, int(self.settings.opencode_max_log_lines))
        if len(result.console_lines) > max_lines:
            del result.console_lines[: len(result.console_lines) - max_lines]

    def _redact_log_line(self, line: str) -> str:
        redacted = str(line or "")
        redacted = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,}]+", r"\1[redacted]", redacted)
        redacted = re.sub(r"(?i)((?:access|refresh|id)?token\s*[=:]\s*)[^\s,}]+", r"\1[redacted]", redacted)
        redacted = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s,}]+", r"\1[redacted]", redacted)
        return redacted
