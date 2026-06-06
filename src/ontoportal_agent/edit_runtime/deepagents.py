from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

try:  # Optional until runtime image dependencies are refreshed.
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover - exercised only in old environments missing the extra package.
    ChatAnthropic = None  # type: ignore[assignment]
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


class _AgenticGraphRagInput(BaseModel):
    query: str = Field(..., description="Ontology or materials-domain question to retrieve evidence for via agentic search")
    top_k: int | None = Field(default=None, description="Optional number of chunks to retrieve per search attempt")
    ontology_id: str | None = Field(default=None, description="Optional target ontology ID to restrict the search to")
    strict_scope: bool = Field(default=True, description="Strictly limit search to target ontology if ontology_id is provided")
    allow_scope_expansion: bool = Field(default=False, description="Allow query rewriting to look outside target ontology if initial search fails")
    max_iterations: int = Field(default=3, description="Maximum iterations for rewriting and retrying unsatisfied queries")


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
        with self._maybe_antigravity_proxy(result=result):
            model = self._build_fast_ask_model()
            final_text = yield from self._run_fast_ask_with_model(request=request, result=result, model=model)
        return final_text

    def _run_fast_ask_with_model(self, *, request: EditRuntimeRequest, result: OpenCodeExecutionResult, model: Any) -> Iterator[str]:
        citations = "\n".join(
            f"- {str(label).strip()}" for label in request.citation_labels if str(label or "").strip()
        )
        instructions = self._contextual_ask_instructions(request.context)
        prompt_parts = instructions + [
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
        prompt = "\n".join(prompt_parts)
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

        with self._maybe_antigravity_proxy(result=result):
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

    @contextmanager
    def _maybe_antigravity_proxy(self, *, result: OpenCodeExecutionResult) -> Iterator[None]:
        if not self._uses_antigravity_account_auth():
            yield
            return

        with tempfile.TemporaryDirectory(prefix="matportal-antigravity-proxy-") as temp_dir:
            home = Path(temp_dir)
            accounts_path = home / ".config" / "antigravity-proxy" / "accounts.json"
            accounts_path.parent.mkdir(parents=True, exist_ok=True)
            accounts_path.write_text(
                json.dumps(self._antigravity_proxy_accounts_config(), separators=(",", ":"), ensure_ascii=False),
                encoding="utf-8",
            )
            accounts_path.chmod(0o600)

            port = self._select_antigravity_proxy_port()
            env = dict(os.environ)
            env.update({"HOME": str(home), "PORT": str(port), "HOST": "127.0.0.1"})
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    ["antigravity-claude-proxy", "start"],
                    cwd=str(home),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                base_url = f"http://127.0.0.1:{port}"
                self._wait_for_antigravity_proxy(base_url=base_url, process=process)
                self._active_antigravity_proxy_base_url = base_url
                self._append_console_line(result, f"Deep Agents Antigravity proxy ready on localhost:{port}.")
                yield
            except FileNotFoundError as exc:
                raise DeepAgentsRuntimeError(
                    "Deep Agents Antigravity account auth requires antigravity-claude-proxy in the runtime image."
                ) from exc
            finally:
                if hasattr(self, "_active_antigravity_proxy_base_url"):
                    delattr(self, "_active_antigravity_proxy_base_url")
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

    def _antigravity_proxy_accounts_config(self) -> dict[str, Any]:
        raw_auth = str(getattr(self.account_auth, "opencode_auth_json", "") or "").strip()
        if not raw_auth:
            raise DeepAgentsRuntimeError("Deep Agents Antigravity account auth requires saved Gemini Antigravity OAuth JSON.")
        try:
            auth = json.loads(raw_auth)
        except json.JSONDecodeError as exc:
            raise DeepAgentsRuntimeError("Saved Gemini Antigravity OAuth JSON is invalid.") from exc
        google = auth.get("google") if isinstance(auth, dict) else None
        if not isinstance(google, dict):
            raise DeepAgentsRuntimeError("Saved Gemini Antigravity OAuth JSON is missing the google account object.")
        refresh_token = str(google.get("refresh") or "").strip()
        if not refresh_token:
            raise DeepAgentsRuntimeError("Saved Gemini Antigravity OAuth JSON is missing a refresh token.")
        email = str(google.get("email") or "matportal-antigravity@local").strip() or "matportal-antigravity@local"
        project_id = str(google.get("projectId") or "").strip()
        return {
            "accounts": [
                {
                    "email": email,
                    "source": "oauth",
                    "enabled": True,
                    "refreshToken": refresh_token,
                    "projectId": project_id or None,
                    "addedAt": datetime.now(timezone.utc).isoformat(),
                    "modelRateLimits": {},
                    "lastUsed": None,
                }
            ],
            "settings": {},
            "activeIndex": 0,
        }

    def _wait_for_antigravity_proxy(self, *, base_url: str, process: subprocess.Popen[str]) -> None:
        deadline = time.monotonic() + 20
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise DeepAgentsRuntimeError("Deep Agents Antigravity proxy exited before it became ready.")
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code < 500:
                    return
                last_error = f"status {response.status_code}"
            except requests.RequestException as exc:
                last_error = exc.__class__.__name__
            time.sleep(0.25)
        raise DeepAgentsRuntimeError(f"Deep Agents Antigravity proxy did not become ready ({last_error or 'timeout'}).")

    def _select_antigravity_proxy_port(self) -> int:
        preferred = self._configured_antigravity_proxy_port()
        if preferred and self._port_is_available(preferred):
            return preferred
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _configured_antigravity_proxy_port(self) -> int:
        parsed = urlparse(str(self.settings.deepagents_antigravity_base_url or "http://localhost:51200/v1"))
        try:
            return int(parsed.port or 51200)
        except ValueError:
            return 51200

    def _configured_antigravity_proxy_base_url(self) -> str:
        parsed = urlparse(str(self.settings.deepagents_antigravity_base_url or "http://localhost:51200/v1"))
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "localhost"
        port = parsed.port or 51200
        return f"{scheme}://{host}:{port}"

    def _port_is_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", int(port)))
            except OSError:
                return False
        return True

    def _build_antigravity_account_model(self, *, model_override: str | None = None) -> Any:
        if ChatAnthropic is None:
            raise DeepAgentsRuntimeError(
                "Deep Agents Antigravity account auth requires the langchain-anthropic package in the runtime image."
            )
        model = self._antigravity_proxy_model_id(model_override=model_override)
        base_url = getattr(self, "_active_antigravity_proxy_base_url", None) or self._configured_antigravity_proxy_base_url()
        return ChatAnthropic(
            api_key=str(self.settings.deepagents_antigravity_api_key or "proxy-managed"),
            base_url=base_url,
            model=model,
            temperature=0.0,
            max_tokens=4096,
        )

    def _antigravity_proxy_model_id(self, *, model_override: str | None = None) -> str:
        configured = str(model_override or "").strip()
        if configured:
            return self._normalize_antigravity_proxy_model_id(configured.split("/", 1)[-1])
        raw = str(getattr(self.account_auth, "model_ref", "") or "").strip()
        model_id = raw.split("/", 1)[-1] if raw else ""
        return self._normalize_antigravity_proxy_model_id(model_id)

    def _normalize_antigravity_proxy_model_id(self, model_id: str) -> str:
        normalized = str(model_id or "").strip().replace("antigravity-", "")
        lower = normalized.lower()
        if not lower or lower in {"gemini-3-pro", "gemini-3.1-pro"}:
            return "gemini-3.1-pro-high"
        if lower in {"gemini-3-pro-preview", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools"}:
            return "gemini-3.1-pro-low"
        if "flash-lite" in lower or lower in {"gemini-3-flash-preview", "gemini-3.1-flash-preview"}:
            return "gemini-3.5-flash-low"
        if lower == "gemini-3-flash":
            return "gemini-3-flash"
        return normalized

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

        def agentic_graphrag_sufficient_evidence(
            query: str,
            top_k: int | None = None,
            ontology_id: str | None = None,
            strict_scope: bool = True,
            allow_scope_expansion: bool = False,
            max_iterations: int = 3,
        ) -> dict[str, Any]:
            options = self.runtime_options
            base_url = str(getattr(options, "rag_base_url", "") or self.settings.rag_base_url or "")
            query_path = str(getattr(options, "rag_query_path", "") or self.settings.rag_query_path or "")
            try:
                model = self.model or self._build_model()
                client = RagClient(base_url=base_url, query_path=query_path)
                from ..agentic_graphrag import run_agentic_graphrag
                return run_agentic_graphrag(
                    question=query,
                    rag_client=client,
                    llm=model,
                    ontology_id=ontology_id,
                    strict_scope=strict_scope,
                    allow_scope_expansion=allow_scope_expansion,
                    max_iterations=max_iterations,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001 - model needs unavailable evidence recorded, not a crash.
                return {
                    "sufficient_context": False,
                    "coverage": [],
                    "gaps": [query],
                    "attempts": [],
                    "sources": [],
                    "final_context": "",
                    "error": str(exc),
                }

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
                func=agentic_graphrag_sufficient_evidence,
                name="matportal_graphrag_sufficient_evidence",
                description=(
                    "Retrieve MatPortal ontology evidence using a bounded agentic GraphRAG loop. "
                    "This tool decomposes the question into atomic needs, iteratively queries and rewrites them "
                    "up to max_iterations. Preserves guardrails: no raw text-to-SPARQL is performed, no silent "
                    "scope expansion occurs, all claims are tied to returned evidence source records, and unresolved "
                    "needs are reported as explicit gaps."
                ),
                args_schema=_AgenticGraphRagInput,
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
                "Use matportal_graphrag_sufficient_evidence for complex, multi-hop, or term reuse questions to perform a bounded agentic retrieval loop, and record its coverage/gaps in evidence-ledger.json.",
                "Use matportal_rag_query before drafting when simpler semantic evidence is needed.",
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

    def _contextual_ask_instructions(self, context: dict[str, Any] | None) -> list[str]:
        if not context:
            return [
                "You are answering a MatPortal assistant Ask request.",
                "Answer fast. Do not deliberate at length. Do not use tools. Do not create files.",
                "For basic portal-help questions, give a short direct product explanation first.",
                "Use the retrieved context for specifics; if it is weak or missing, say what is missing.",
                "Keep the answer concise, direct, and operator-facing."
            ]

        kind = str(context.get("page_kind") or "").strip().lower()
        instructions = [
            f"You are a specialized MatPortal assistant for page_kind={kind.upper()}.",
            "Answer fast and direct. Do not deliberate. Do not create files unless explicitly requested."
        ]

        if kind == "general":
            instructions.extend([
                "Your focus is general MatPortal portal help.",
                "When the user asks what MatPortal is, explain that it is a materials-science ontology portal built on OntoPortal for browsing, searching, annotating, mapping, and sharing ontologies.",
                "Answer in 2-4 sentences unless the user asks for more detail."
            ])
        elif kind == "search":
            q = context.get("search_query")
            instructions.extend([
                "Your focus is Search helper.",
                "Help the user refine their search, select appropriate ontologies, or recommend search terms.",
                f"Active search context: query={q or '(none)'}."
            ])
        elif kind == "ontology":
            acr = context.get("ontology_acronym")
            name = context.get("ontology_name")
            instructions.extend([
                f"Your focus is the Ontology summary for: {name} ({acr}).",
                "Answer questions about this ontology, its metadata, domain, metrics, or import structure."
            ])
        elif kind == "concept":
            acr = context.get("ontology_acronym")
            cid = context.get("concept_id")
            clabel = context.get("concept_label")
            instructions.extend([
                f"Your focus is the Class details for: {clabel or cid} inside ontology {acr}.",
                "Help explain this class, recommend parents/children, suggest definition improvements, or map synonyms."
            ])
        elif kind == "mappings":
            instructions.extend([
                "Your focus is Mappings helper.",
                "Help the user define mappings, write mapping rationales, and verify equivalence relations."
            ])
        elif kind == "sparql":
            instructions.extend([
                "Your focus is SPARQL Query helper.",
                "Write, debug, and explain SPARQL queries for the current ontology or triple store."
            ])
        elif kind == "recommender":
            instructions.extend([
                "Your focus is Recommender helper.",
                "Suggest keywords or text structures that produce optimal recommender rankings."
            ])
        elif kind == "annotator":
            instructions.extend([
                "Your focus is Annotator helper.",
                "Help configure annotator parameters and explain text annotation outputs."
            ])

        instructions.append("Use the retrieved context below for specifics; if it is weak or missing, answer basic portal-help questions directly and note when deployment-specific details may vary.")
        return instructions
