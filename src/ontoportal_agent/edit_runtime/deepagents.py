from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..agent.options import AgentRuntimeOptions
from ..artifact_store import ArtifactAccessError, assert_artifact_safe_for_exposure, sanitize_artifact_path
from ..config import AgentSettings, get_settings
from ..mcp_client import McpClient
from ..opencode_executor import OpenCodeAccountAuth, OpenCodeExecutionResult, OpenCodeExecutor
from ..rag_client import RagClient
from .base import EditRuntimeCapabilities, EditRuntimeRequest


class DeepAgentsRuntimeError(RuntimeError):
    status_code = 503


class _ArtifactWriteInput(BaseModel):
    path: str = Field(..., description="Relative artifact path inside the assistant workspace")
    content: str = Field(..., description="Complete UTF-8 text content to write")


class _ArtifactReadInput(BaseModel):
    path: str = Field(..., description="Relative artifact path inside the assistant workspace")


class _RagQueryInput(BaseModel):
    query: str = Field(..., description="Ontology or materials-domain question to retrieve evidence for")
    top_k: int | None = Field(default=None, description="Optional number of chunks to retrieve")


class _McpInvokeInput(BaseModel):
    tool_name: str = Field(..., description="Name of the configured MCP tool to invoke")
    arguments_json: str = Field(default="{}", description="JSON object string containing tool arguments")


class DeepAgentsEditRuntime:
    """LangChain Deep Agents based edit runtime for ontology workspaces.

    This adapter deliberately reuses the existing OpenCode workspace/finalization
    boundary so artifacts, diffs, validation, expiry, and session persistence keep
    the same backend/UI contract while the agent loop can be swapped.
    """

    capabilities = EditRuntimeCapabilities(
        runtime="deepagents",
        supports_sessions=False,
        supports_cancel=False,
        supports_mcp=True,
        supports_artifacts=True,
    )

    runtime = "deepagents"

    def __init__(
        self,
        *,
        settings: AgentSettings | None = None,
        runtime_options: AgentRuntimeOptions | None = None,
        mcp_servers: list[dict[str, Any] | str] | None = None,
        account_auth: OpenCodeAccountAuth | None = None,
        model: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.runtime_options = runtime_options
        self.mcp_servers = list(mcp_servers or [])
        self.account_auth = account_auth
        self.model = model
        self._workspace_manager = OpenCodeExecutor(settings=self.settings, mcp_servers=self.mcp_servers)
        self._mcp_client = McpClient(
            self.mcp_servers or getattr(self.runtime_options, "mcp_endpoints", []) or self.settings.resolved_mcp_endpoints(),
            api_key=getattr(self.runtime_options, "mcp_api_key", None) or self.settings.mcp_api_key,
        )

    def stream(self, request: EditRuntimeRequest) -> Iterator[dict[str, Any]]:
        if not self.settings.deepagents_enabled:
            raise DeepAgentsRuntimeError("Deep Agents edit runtime is disabled by ONTOAGENT_DEEPAGENTS_ENABLED.")
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
                "title": "Deep Agents workspace",
            },
        }
        yield {"type": "opencode_phase", "content": {"label": "Preparing Deep Agents workspace", "run_id": run_id, "workspace": str(workspace), "runtime": self.runtime}}

        try:
            if str(request.task or "").strip().lower() == "ask":
                final_text = yield from self._run_fast_ask(request=request, result=result)
            else:
                final_text = yield from self._run_deep_agent(request=request, workspace=workspace, result=result)
            result.final_text = final_text.strip() or "Deep Agents completed the request."
            result.exit_code = 0
        except Exception as exc:  # noqa: BLE001 - convert runtime failures to the established result contract.
            result.exit_code = int(getattr(exc, "status_code", 1) or 1)
            result.final_text = str(exc) or exc.__class__.__name__
            self._append_console_line(result, f"Deep Agents failed: {result.final_text}")
            yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}

        self._workspace_manager._finalize_workspace(workspace=workspace, result=result)
        self._workspace_manager._classify_execution_result(result=result, task=request.task)
        if result.failure_reason:
            self._append_console_line(result, f"Deep Agents result rejected: {result.failure_reason}")
            yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        yield {"type": "changed_files", "content": result.changed_files}
        yield {"type": "diff_summary", "content": result.diff_summary}
        yield {"type": "artifact_candidates", "content": result.artifact_candidates}
        yield {"type": "validation_report", "content": result.validation_report}

        if result.ok:
            yield {"type": "opencode_phase", "content": {"label": "Deep Agents workspace complete", "workspace": str(workspace), "runtime": self.runtime}}
        else:
            yield {
                "type": "opencode_phase",
                "content": {
                    "label": "Deep Agents workspace failed",
                    "workspace": str(workspace),
                    "runtime": self.runtime,
                    "exit_code": result.exit_code,
                    "failure_kind": result.failure_kind,
                },
            }
        return result

    def _run_fast_ask(self, *, request: EditRuntimeRequest, result: OpenCodeExecutionResult) -> Iterator[str]:
        model = self._build_fast_ask_model()
        citations = "\n".join(
            f"- {str(label).strip()}" for label in request.citation_labels if str(label or "").strip()
        )
        prompt = "\n".join(
            [
                "You are answering a MatPortal assistant Ask request.",
                "Answer fast. Do not deliberate at length. Do not use tools. Do not create files.",
                "Use only the retrieved context below; if it is weak or missing, say what is missing.",
                "Keep the answer concise, direct, and operator-facing.",
                "",
                "User question:",
                request.prompt.strip(),
                "",
                "Retrieved context:",
                str(request.retrieved_context or "").strip() or "(none)",
                "",
                "Citations:",
                citations or "- none",
                "",
                "Return only the answer text.",
            ]
        )
        self._append_console_line(result, "Deep Agents fast Ask generation started.")
        yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        reply = model.invoke([HumanMessage(content=prompt)])
        self._append_console_line(result, "Deep Agents fast Ask generation finished.")
        yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        return self._stringify_content(getattr(reply, "content", reply))

    def _run_deep_agent(self, *, request: EditRuntimeRequest, workspace: Path, result: OpenCodeExecutionResult) -> Iterator[str]:
        try:
            from deepagents import create_deep_agent
            from deepagents.backends import StateBackend
            from deepagents.middleware import FilesystemPermission
        except Exception as exc:  # pragma: no cover - covered in integration environments with optional dep absent.
            raise DeepAgentsRuntimeError("Python package deepagents is not installed in the assistant runtime.") from exc

        tools = self._tools(workspace=workspace, result=result)
        model = self.model or self._build_model()
        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=self._system_prompt(workspace),
            backend=StateBackend(),
            permissions=[FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny")],
            interrupt_on=None,
            debug=False,
            name="matportal-deepagents-edit",
        )
        prompt = self._user_prompt(request)
        self._append_console_line(result, "Deep Agents run started.")
        yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        final_state = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        final_text = self._extract_final_text(final_state)
        self._append_console_line(result, "Deep Agents run finished.")
        yield {"type": "terminal_log", "content": {"line": result.console_lines[-1]}}
        return final_text

    def _build_model(self) -> ChatOpenAI:
        return self._build_chat_model(model_override=self.settings.deepagents_model)

    def _build_fast_ask_model(self) -> ChatOpenAI:
        if self.model is not None:
            return self.model
        return self._build_chat_model(model_override=self.settings.ask_runtime_model or self.settings.deepagents_model)

    def _build_chat_model(self, *, model_override: str | None = None) -> ChatOpenAI:
        options = self.runtime_options
        if self._uses_antigravity_account_auth():
            return self._build_antigravity_account_model(model_override=model_override)
        api_key = str(getattr(options, "openai_api_key", "") or self.settings.openai_api_key or "")
        base_url = str(getattr(options, "openai_api_base", "") or self.settings.openai_api_base or "") or None
        model = str(model_override or getattr(options, "llm_model", "") or self.settings.llm_model or "").strip()
        if not base_url and model.lower().startswith("gemini"):
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        if not api_key:
            raise DeepAgentsRuntimeError("Deep Agents runtime requires a configured generation API key.")
        if not model:
            raise DeepAgentsRuntimeError("Deep Agents runtime requires a configured model.")
        return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0.0)

    def _uses_antigravity_account_auth(self) -> bool:
        return bool(self.account_auth and str(self.account_auth.kind or "").strip().lower() == "gemini_antigravity")

    def _build_antigravity_account_model(self, *, model_override: str | None = None) -> ChatOpenAI:
        model = self._antigravity_proxy_model_id(model_override=model_override)
        return ChatOpenAI(
            api_key=str(self.settings.deepagents_antigravity_api_key or "proxy-managed"),
            base_url=str(self.settings.deepagents_antigravity_base_url or "http://localhost:51200/v1"),
            model=model,
            temperature=0.0,
        )

    def _antigravity_proxy_model_id(self, *, model_override: str | None = None) -> str:
        configured = str(model_override or "").strip()
        if configured:
            return configured.split("/", 1)[-1].replace("antigravity-", "")
        raw = str(getattr(self.account_auth, "model_ref", "") or "").strip()
        model_id = raw.split("/", 1)[-1] if raw else ""
        model_id = model_id.replace("antigravity-", "")
        if model_id in {"gemini-3-pro", "gemini-3.1-pro", ""}:
            return "gemini-3.1-pro-high"
        return model_id

    def _model_label(self) -> str:
        if self._uses_antigravity_account_auth():
            return f"deepagents/antigravity/{self._antigravity_proxy_model_id(model_override=self.settings.deepagents_model)}"
        configured = str(self.settings.deepagents_model or getattr(self.runtime_options, "llm_model", "") or self.settings.llm_model or "").strip()
        return f"deepagents/{configured or 'default'}"

    def _tools(self, *, workspace: Path, result: OpenCodeExecutionResult) -> list[StructuredTool]:
        def write_artifact(path: str, content: str) -> dict[str, Any]:
            safe = sanitize_artifact_path(path)
            target = (workspace / safe).resolve()
            try:
                target.relative_to(workspace.resolve())
            except ValueError as exc:
                raise ArtifactAccessError("Artifact path escapes the Deep Agents workspace.") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content or ""), encoding="utf-8")
            try:
                assert_artifact_safe_for_exposure(target)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return {"path": safe.as_posix(), "bytes": target.stat().st_size, "status": "written"}

        def read_artifact(path: str) -> dict[str, Any]:
            safe = sanitize_artifact_path(path)
            target = (workspace / safe).resolve()
            try:
                target.relative_to(workspace.resolve())
            except ValueError as exc:
                raise ArtifactAccessError("Artifact path escapes the Deep Agents workspace.") from exc
            if not target.is_file():
                return {"path": safe.as_posix(), "error": "not_found"}
            return {"path": safe.as_posix(), "content": target.read_text(encoding="utf-8", errors="replace")[:120_000]}

        def rag_query(query: str, top_k: int | None = None) -> dict[str, Any]:
            options = self.runtime_options
            base_url = str(getattr(options, "rag_base_url", "") or self.settings.rag_base_url or "")
            query_path = str(getattr(options, "rag_query_path", "") or self.settings.rag_query_path or "")
            rag_top_k = int(top_k or getattr(options, "rag_top_k", 0) or 10)
            try:
                rag_result = RagClient(base_url=base_url, query_path=query_path).query(query, top_k=rag_top_k)
                return {
                    "answer": rag_result.answer,
                    "sources": [source.__dict__ for source in rag_result.sources],
                }
            except Exception as exc:  # noqa: BLE001 - model needs unavailable evidence recorded, not a crash.
                return {"answer": "", "sources": [], "error": str(exc)}

        def invoke_mcp_tool(tool_name: str, arguments_json: str = "{}") -> dict[str, Any]:
            try:
                arguments = json.loads(arguments_json or "{}")
            except json.JSONDecodeError as exc:
                return {"error": f"arguments_json must be a JSON object: {exc}"}
            if not isinstance(arguments, dict):
                return {"error": "arguments_json must decode to a JSON object"}
            try:
                return self._mcp_client.invoke(tool_name, arguments)
            except Exception as exc:  # noqa: BLE001 - failed evidence tools should be captured in artifacts.
                return {"error": str(exc), "tool_name": tool_name}

        def validate_workspace() -> dict[str, Any]:
            self._refresh_workspace_report(workspace=workspace, result=result)
            return result.validation_report

        return [
            StructuredTool.from_function(
                func=write_artifact,
                name="matportal_write_artifact",
                description="Write a proposal, evidence, report, validation, or draft artifact under the assistant workspace.",
                args_schema=_ArtifactWriteInput,
            ),
            StructuredTool.from_function(
                func=read_artifact,
                name="matportal_read_artifact",
                description="Read a previously written workspace artifact by relative path.",
                args_schema=_ArtifactReadInput,
            ),
            StructuredTool.from_function(
                func=rag_query,
                name="matportal_rag_query",
                description="Retrieve MatPortal ontology evidence and source chunks before drafting ontology edits.",
                args_schema=_RagQueryInput,
            ),
            StructuredTool.from_function(
                func=invoke_mcp_tool,
                name="matportal_mcp_invoke",
                description="Invoke a configured MatPortal MCP tool such as ontoportal_api for exact ontology/API evidence.",
                args_schema=_McpInvokeInput,
            ),
            StructuredTool.from_function(
                func=validate_workspace,
                name="matportal_validate_workspace",
                description="Run the MatPortal artifact validators over current workspace changes.",
            ),
        ]

    def _refresh_workspace_report(self, *, workspace: Path, result: OpenCodeExecutionResult) -> None:
        subprocess.run(["git", "add", "-A"], cwd=str(workspace), check=False, capture_output=True, text=True)
        status_output = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        result.changed_files = self._workspace_manager._parse_changed_files(status_output, workspace=workspace)
        result.artifact_candidates = self._workspace_manager._artifact_candidates(result.changed_files)
        result.validation_report = self._workspace_manager._build_validation_report(
            workspace=workspace,
            changed_files=result.changed_files,
        )

    def _system_prompt(self, workspace: Path) -> str:
        return "\n".join(
            [
                "You are the MatPortal ontology edit runtime running inside LangChain Deep Agents.",
                "You must prepare reviewable ontology edit artifacts, not publish live data.",
                f"Workspace root: {workspace}",
                "Use matportal_rag_query before drafting when semantic evidence is needed.",
                "Use matportal_mcp_invoke for exact OntoPortal API/MCP evidence when configured, and record unavailability if it fails.",
                "Write files only through matportal_write_artifact; do not rely on built-in filesystem scratch files for deliverables.",
                "Required artifacts for edit tasks: edit-plan.json, evidence-ledger.json, operator-report.md, validation-summary.json, draft-submission.md, and at least one ontology artifact when a content change is requested.",
                "Call matportal_validate_workspace after writing ontology artifacts and reflect the result in validation-summary.json.",
                "Never include secrets, API keys, OAuth tokens, local absolute paths, or runtime config in artifacts.",
                "Do not publish, commit, push, modify remotes, or mutate live OntoPortal data.",
            ]
        )

    def _user_prompt(self, request: EditRuntimeRequest) -> str:
        if request.task == "ask":
            return request.prompt
        return "\n".join(
            [
                "User ontology edit request:",
                request.prompt,
                "",
                "Return a concise operator summary after all required files are written and validation has run.",
            ]
        )

    def _extract_final_text(self, final_state: Any) -> str:
        if isinstance(final_state, dict):
            messages = final_state.get("messages") or []
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    return self._stringify_content(message.content)
                if getattr(message, "type", None) == "ai" or message.__class__.__name__ == "AIMessage":
                    return self._stringify_content(getattr(message, "content", ""))
            structured = final_state.get("structured_response")
            if structured is not None:
                return json.dumps(structured, ensure_ascii=False, default=str)
        return str(final_state or "")

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
        result.console_lines.append(line)
        max_lines = max(1, int(self.settings.opencode_max_log_lines))
        if len(result.console_lines) > max_lines:
            del result.console_lines[: len(result.console_lines) - max_lines]
