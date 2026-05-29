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

from .antigravity_models import antigravity_opencode_provider_config, normalize_antigravity_model_ref
from .artifact_store import resolve_safe_workspace
from .config import AgentSettings, get_settings
from .ontology.diagnostics import normalize_validation_report
from .ontology.proposals import STRUCTURED_ONTOLOGY_ARTIFACTS, validate_ontology_proposal_payload

_ONTOLOGY_ARTIFACT_SUFFIXES = {".ttl", ".rdf", ".owl", ".json", ".yaml", ".yml", ".md", ".txt"}
_WORKFLOW_REQUIRED_ARTIFACTS = (
    "edit-plan.json",
    "evidence-ledger.json",
    "operator-report.md",
    "validation-summary.json",
    "draft-submission.md",
)
_WORKFLOW_ONTOLOGY_SUFFIXES = {".ttl", ".rdf", ".owl"}
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
_BLOCKED_BASH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|\s)(apt|apt-get)\s+install(\s|$)", re.IGNORECASE), "system package installation is blocked"),
    (re.compile(r"(^|\s)(dnf|yum|apk|pacman|zypper)\s+install(\s|$)", re.IGNORECASE), "system package installation is blocked"),
    (re.compile(r"(^|\s)(pip|pip3)\s+install(\s|$)", re.IGNORECASE), "python package installation is blocked"),
    (re.compile(r"(^|\s)npm\s+install(\s|$)", re.IGNORECASE), "node package installation is blocked"),
    (re.compile(r"(^|\s)(curl|wget)\b[^\n]*\|\s*(sh|bash)\b", re.IGNORECASE), "remote shell execution is blocked"),
    (re.compile(r"(^|\s)sudo(\s|$)", re.IGNORECASE), "privilege escalation is blocked"),
    (re.compile(r"(^|\s)(shutdown|reboot|poweroff|halt)(\s|$)", re.IGNORECASE), "host power control is blocked"),
    (re.compile(r"(^|\s)(mkfs|fdisk|parted)(\s|$)", re.IGNORECASE), "disk formatting commands are blocked"),
)
_USER_PROVIDER_ID = "matportal-user"
_USER_PROVIDER_API_KEY_ENV = "MATPORTAL_OPENCODE_API_KEY"
_OPENAI_COMPATIBLE_NPM = "@ai-sdk/openai-compatible"
_ONTOLOGY_TOOLKIT_DIR = "matportal-ontology-toolkit"
_ANTIGRAVITY_AUTH_KIND = "gemini_antigravity"
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


@dataclass(frozen=True)
class OpenCodeAccountAuth:
    kind: str
    opencode_auth_json: str | None = field(default=None, repr=False)
    codex_auth_json: str | None = field(default=None, repr=False)
    model_ref: str | None = None


@dataclass
class OpenCodeExecutionResult:
    ok: bool
    workspace: str
    run_id: str
    expires_at: str
    session_id: str | None = None
    model: str | None = None
    runtime: str = "opencode"
    final_text: str = ""
    exit_code: int = 0
    timed_out: bool = False
    blocked: bool = False
    blocked_reason: str = ""
    console_lines: list[str] = field(default_factory=list)
    changed_files: list[dict[str, Any]] = field(default_factory=list)
    diff_summary: dict[str, Any] = field(default_factory=dict)
    artifact_candidates: list[dict[str, Any]] = field(default_factory=list)
    validation_report: dict[str, Any] = field(default_factory=dict)

    def execution_payload(self) -> dict[str, Any]:
        return {
            "mode": self.runtime,
            "ok": self.ok,
            "run_id": self.run_id,
            "workspace": self.workspace,
            "session_id": self.session_id,
            "model": self.model,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "logs": self.console_lines,
            "changed_files": self.changed_files,
            "diff_summary": self.diff_summary,
            "artifact_candidates": self.artifact_candidates,
            "validation_report": self.validation_report,
            "expires_at": self.expires_at,
        }


class OpenCodeExecutor:
    def __init__(
        self,
        settings: AgentSettings | None = None,
        provider_auth: OpenCodeProviderAuth | None = None,
        account_auth: OpenCodeAccountAuth | None = None,
        mcp_servers: list[str | dict[str, Any]] | None = None,
    ):
        self.settings = settings or get_settings()
        self.provider_auth = provider_auth
        self.account_auth = account_auth
        self.mcp_servers = list(mcp_servers or [])
        self._runtime_mcp_secret_env: dict[str, str] = {}

    def stream(
        self,
        *,
        prompt: str,
        thread_id: str | None,
        trace_id: str | None = None,
        task: str = "edit",
        retrieved_context: str = "",
        citation_labels: list[str] | None = None,
        resume_workspace: str | None = None,
        resume_session_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        run_id = uuid.uuid4().hex
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=max(1, int(self.settings.opencode_artifact_retention_days)))
        ).isoformat()
        workspace = self._prepare_workspace(
            thread_id=thread_id,
            run_id=run_id,
            resume_workspace=resume_workspace,
        )
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
            session_id=resume_session_id,
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

        if result.exit_code == 0:
            result.ok = True
            yield {
                "type": "opencode_phase",
                "content": {
                    "label": "Workspace complete",
                    "workspace": str(workspace),
                },
            }
        else:
            self._append_console_line(result, f"OpenCode exited with code {result.exit_code}.")
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

    def _prepare_workspace(
        self,
        *,
        thread_id: str | None,
        run_id: str | None = None,
        resume_workspace: str | None = None,
    ) -> Path:
        root = (self.settings.ontology_workdir / self.settings.opencode_workspace_subdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._chmod_private(root, 0o700)
        workspace: Path
        if resume_workspace:
            workspace = resolve_safe_workspace(root, resume_workspace)
        elif thread_id:
            token = str(thread_id).replace("/", "-")
            workspace = root / token
        else:
            token = "standalone"
            workspace = root / f"{token}-{run_id or int(time.time())}"
        is_new_workspace = not workspace.exists()
        workspace.mkdir(parents=True, exist_ok=True)
        self._chmod_private(workspace, 0o700)
        self._runtime_mcp_secret_env = {}

        config = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": self._opencode_mcp_configs(),
        }
        plugins = self._opencode_plugins()
        if plugins:
            config["plugin"] = plugins
        permissions = self._opencode_permissions()
        if permissions:
            config["permission"] = permissions
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
        elif self._uses_antigravity_account_auth():
            config["provider"] = antigravity_opencode_provider_config()
        config_path = workspace / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._chmod_private(config_path, 0o600)
        (workspace / "README.md").write_text(
            "\n".join(
                [
                    "# OpenCode Ontology Workspace",
                    "",
                    "This workspace is disposable.",
                    "- Use the matportal_rag MCP server for semantic retrieval and source chunks.",
                    "- Use the ontoportal_api MCP server for exact ontology/API state and full ontology access.",
                    "- If Antigravity auth is active, use google_search for domain research with citations.",
                    f"- Use `{_ONTOLOGY_TOOLKIT_DIR}/` for proposal templates and review checklists.",
                    "- Write proposed ontology changes into files under this directory and prepare a draft submission bundle.",
                    "- Copy toolkit templates into new proposal files; do not edit toolkit files unless asked.",
                    "- Do not commit, push, or modify remotes.",
                ]
            ),
            encoding="utf-8",
        )
        self._write_ontology_toolkit(workspace)
        if is_new_workspace or not (workspace / ".git").exists():
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
            "ontology-proposal-template.json": self._ontology_proposal_template(),
            "competency-questions-template.json": self._competency_questions_template(),
            "reuse-candidates-template.json": self._reuse_candidates_template(),
            "validation-summary-template.json": self._validation_summary_template(),
            "operator-report-template.md": self._operator_report_template(),
            "review-checklist.json": self._review_checklist_template(),
            "draft-submission-template.md": self._draft_submission_template(),
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
                "1. Use `matportal_rag` first to retrieve source chunks and terminology relevant to the request.",
                "2. Use `ontoportal_api` to inspect exact ontology metadata, terms, classes, submissions, and full ontology files when needed.",
                "3. Use provider web search only when needed for domain modeling; with Antigravity, prefer `google_search` and cite sources.",
                "4. Write an edit plan before drafting, then inspect the ontology again after drafting or validation feedback.",
                "5. Copy `proposal-template.ttl` into a new `.ttl` file for RDF/Turtle proposals.",
                "6. When ontology copilot schema mode is requested, copy the JSON templates to workspace-root files named `ontology-proposal.json`, `competency-questions.json`, `reuse-candidates.json`, and `validation-summary.json`.",
                "7. Treat reuse as mandatory evidence: record candidate terms, why they were reused/mapped/extended/rejected, and cite RAG/API/web evidence.",
                "8. Prefer SKOS mapping relations for external alignments; do not propose `owl:equivalentClass` or `owl:equivalentProperty` unless the operator explicitly requests and reviews it.",
                "9. Copy `operator-report-template.md` and `draft-submission-template.md` for review notes and a draft submission package.",
                "10. Keep generated artifacts at the workspace root or in a purpose-named subdirectory.",
                "11. Finish with a short summary naming changed files, validation status, search/provenance, and assumptions.",
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

    def _ontology_proposal_template(self) -> str:
        return json.dumps(
            {
                "schema_version": "ontology-copilot/v1",
                "proposal_id": "draft-proposal-1",
                "title": "Add reviewable processing-method term",
                "summary": "Proposal-only ontology edit for operator review; no live apply or publish action is requested.",
                "ontology_acronym": "TARGET",
                "goals": ["Answer the competency question without duplicating reusable existing terms."],
                "scope": "In scope: one candidate class and review notes. Out of scope: publishing, imports, inferred hierarchy changes, or unreviewed equivalence axioms.",
                "competency_questions": [
                    {
                        "id": "CQ1",
                        "question": "Which materials or processes are instances/subclasses of the proposed concept?",
                        "expected_answer": "The ontology can retrieve the proposed class, its label, definition/evidence, and parent relation after review.",
                        "status": "draft",
                    }
                ],
                "reuse_candidates": [
                    {
                        "label": "Existing processing method candidate",
                        "iri": "https://example.org/existing-processing-method",
                        "source_ontology": "MatPortal/OntoPortal inspection",
                        "confidence": 0.4,
                        "recommended_action": "needs_review",
                        "rationale": "Record whether to reuse, map with SKOS, extend as a subclass, or reject as out of scope.",
                    }
                ],
                "operations": [
                    {
                        "operation": "create_class",
                        "entity_type": "class",
                        "iri": "https://example.org/target/ProposedProcessingMethod",
                        "label": "Proposed processing method",
                        "parent_iri": "https://example.org/target/ProcessingMethod",
                        "target_iri": "",
                        "mapping_relation": None,
                        "turtle": "",
                        "rationale": "Needed to answer CQ1 after reuse candidates were checked and no exact reusable term was selected.",
                        "evidence": [{"source": "RAG/API inspection", "citation": "chunk-or-endpoint-id", "url": "", "quote": "Short non-secret evidence quote."}],
                    }
                ],
                "assumptions": ["Operator will confirm namespace and parent class before apply/publish work exists."],
                "risks": ["A better reuse candidate may exist; prefer reuse or SKOS mapping if confirmed."],
            },
            indent=2,
        ) + "\n"

    def _competency_questions_template(self) -> str:
        return json.dumps(
            {
                "schema_version": "ontology-copilot/v1",
                "questions": [
                    {
                        "id": "CQ1",
                        "question": "What question should the ontology answer?",
                        "expected_answer": "Describe the expected answer or inference.",
                        "status": "draft",
                    }
                ],
            },
            indent=2,
        ) + "\n"

    def _reuse_candidates_template(self) -> str:
        return json.dumps(
            {
                "schema_version": "ontology-copilot/v1",
                "candidates": [
                    {
                        "label": "Candidate existing term",
                        "iri": "",
                        "source_ontology": "",
                        "confidence": 0.0,
                        "recommended_action": "needs_review",
                        "rationale": "Explain whether to reuse, map, extend, or reject.",
                    }
                ],
            },
            indent=2,
        ) + "\n"

    def _validation_summary_template(self) -> str:
        return json.dumps(
            {
                "schema_version": "ontology-copilot/v1",
                "status": "skipped",
                "summary": "Validation has not run yet.",
                "diagnostics": [],
            },
            indent=2,
        ) + "\n"

    def _operator_report_template(self) -> str:
        return "\n".join(
            [
                "# Operator Review Notes",
                "",
                "## Request",
                "- User request:",
                "",
                "## Inspected Context",
                "- RAG chunks checked:",
                "- OntoPortal API tools/endpoints checked:",
                "- Full ontology files inspected:",
                "- Web sources checked:",
                "",
                "## Edit Plan",
                "-",
                "",
                "## Proposed Artifacts",
                "- Files changed:",
                "- Validation result:",
                "- Draft submission bundle:",
                "",
                "## Provenance",
                "- Ontology/API evidence:",
                "- Web citations:",
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
                    "RAG chunks inspected before drafting",
                    "Exact ontology/API state inspected before drafting",
                    "Ontology state inspected again after drafting or validation",
                    "Domain web research performed when requested or needed",
                    "Web sources are cited in operator notes when used",
                    "Generated RDF parses successfully when applicable",
                    "ROBOT verify/report was run or explicitly marked unavailable",
                    "New terms have labels and definitions where appropriate",
                    "Mappings or external references use stable IRIs",
                    "Draft submission package is present for human review",
                    "Operator notes list assumptions and follow-up actions",
                    "No secrets or absolute local paths are present",
                ]
            },
            indent=2,
        ) + "\n"

    def _draft_submission_template(self) -> str:
        return "\n".join(
            [
                "# Draft MatPortal Submission",
                "",
                "## Target Ontology",
                "- Acronym:",
                "- Latest submission inspected:",
                "- Base ontology/version IRI:",
                "",
                "## Files",
                "- Proposed ontology artifact:",
                "- Operator report:",
                "- Validation report:",
                "",
                "## Submission Metadata",
                "- Version:",
                "- Released:",
                "- Status: draft",
                "- Description:",
                "",
                "## Human Approval Checklist",
                "- Review class/property IRIs and namespace choices.",
                "- Review provenance and citations.",
                "- Review ROBOT report warnings.",
                "- Confirm whether to publish or request revisions.",
                "",
            ]
        )

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
        if self.account_auth:
            self._write_account_auth(workspace, home, data_home)
            codex_home = workspace / ".codex-home"
            codex_home.mkdir(parents=True, exist_ok=True)
            self._chmod_private(codex_home, 0o700)
            env["CODEX_HOME"] = str(codex_home)
        if self.provider_auth:
            env[self.provider_auth.env_api_key_name] = self.provider_auth.api_key
        for key, value in self._runtime_mcp_secret_env.items():
            env[key] = value
        if self.settings.opencode_exa_websearch_enabled:
            env["OPENCODE_ENABLE_EXA"] = "1"
        return env

    def _write_private_json_file(self, path: Path, raw_json: str) -> None:
        parsed = json.loads(raw_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._chmod_private(path.parent, 0o700)
        path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        self._chmod_private(path, 0o600)

    def _write_account_auth(self, workspace: Path, home: Path, data_home: Path) -> None:
        if not self.account_auth:
            return
        if self.account_auth.opencode_auth_json:
            self._write_private_json_file(data_home / "opencode" / "auth.json", self.account_auth.opencode_auth_json)
        if self.account_auth.codex_auth_json:
            codex_home = workspace / ".codex-home"
            self._write_private_json_file(codex_home / "auth.json", self.account_auth.codex_auth_json)

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
                if result.blocked:
                    self._terminate_process_group(process)
                    break

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

    def _opencode_mcp_configs(self) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        seen_urls: set[str] = set()

        def register(name: str, config: dict[str, Any]) -> None:
            clean_name = str(name or "").strip()
            if not clean_name or clean_name in configs:
                return
            url = str(config.get("url") or "").strip()
            if url and url in seen_urls:
                return
            configs[clean_name] = config
            if url:
                seen_urls.add(url)

        register(self.settings.opencode_mcp_name, self._opencode_mcp_config())
        rag_name = str(self.settings.opencode_rag_mcp_name or "").strip()
        if rag_name:
            register(rag_name, self._opencode_rag_mcp_config())
        for index, server in enumerate(self.mcp_servers, start=1):
            resolved = self._runtime_mcp_server_config(server=server, index=index)
            if not resolved:
                continue
            register(resolved["name"], resolved["config"])
        return configs

    def _runtime_mcp_server_config(
        self,
        *,
        server: str | dict[str, Any],
        index: int,
    ) -> dict[str, Any] | None:
        if isinstance(server, str):
            endpoint = str(server).strip()
            if not endpoint:
                return None
            return {
                "name": f"mcp_{index}",
                "config": {
                    "type": "remote",
                    "url": endpoint,
                    "enabled": True,
                    "timeout": self.settings.opencode_rag_mcp_timeout_ms,
                },
            }

        endpoint = str(server.get("url") or "").strip()
        if not endpoint:
            return None
        timeout = server.get("timeout_ms")
        try:
            timeout_ms = int(timeout)
        except (TypeError, ValueError):
            timeout_ms = 0
        if timeout_ms <= 0:
            timeout_ms = self.settings.opencode_rag_mcp_timeout_ms
        name = str(server.get("name") or "").strip() or f"mcp_{index}"
        headers = self._runtime_mcp_headers(server=server, index=index)
        config: dict[str, Any] = {
            "type": "remote",
            "url": endpoint,
            "enabled": True,
            "timeout": timeout_ms,
        }
        if headers:
            config["headers"] = headers
        return {
            "name": name,
            "config": config,
        }

    def _runtime_mcp_headers(self, *, server: dict[str, Any], index: int) -> dict[str, str]:
        literal_headers: dict[str, str] = {}
        raw_headers = server.get("headers")
        if isinstance(raw_headers, dict):
            for key, value in raw_headers.items():
                header_name = str(key or "").strip()
                if not header_name:
                    continue
                header_value = str(value or "").strip()
                if not header_value:
                    continue
                literal_headers[header_name] = header_value
        api_key = str(server.get("api_key") or "").strip()
        if api_key and not any(key.lower() == "x-api-key" for key in literal_headers):
            literal_headers["X-API-Key"] = api_key

        rendered_headers: dict[str, str] = {}
        for header_name, value in literal_headers.items():
            env_name = self._runtime_mcp_env_var_name(index=index, header_name=header_name)
            self._runtime_mcp_secret_env[env_name] = value
            rendered_headers[header_name] = f"{{env:{env_name}}}"
        return rendered_headers

    def _runtime_mcp_env_var_name(self, *, index: int, header_name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9]", "_", header_name.strip().upper()).strip("_")
        if not safe_name:
            safe_name = "HEADER"
        return f"MATPORTAL_MCP_{index}_{safe_name}"

    def _opencode_plugins(self) -> list[str]:
        if self._uses_antigravity_account_auth():
            plugin = str(self.settings.opencode_antigravity_plugin or "").strip()
            return [plugin] if plugin else []
        return []

    def _opencode_permissions(self) -> dict[str, str]:
        if self.settings.opencode_exa_websearch_enabled:
            return {
                "websearch": "allow",
                "webfetch": "allow",
            }
        return {}

    def _uses_antigravity_account_auth(self) -> bool:
        return bool(self.account_auth and str(self.account_auth.kind or "").strip().lower() == _ANTIGRAVITY_AUTH_KIND)

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

    def _opencode_rag_mcp_config(self) -> dict[str, Any]:
        return {
            "type": "remote",
            "url": self._opencode_rag_mcp_url(),
            "enabled": True,
            "timeout": self.settings.opencode_rag_mcp_timeout_ms,
        }

    def _opencode_rag_mcp_url(self) -> str:
        return str(self.settings.opencode_rag_mcp_url or "").strip() or self.settings.rag_base_url.rstrip("/") + "/mcp"

    def _opencode_mcp_url(self) -> str:
        query = urlencode(
            {
                "api_key": self.settings.ontoportal_api_key,
                "base_url": self.settings.ontoportal_api_base.rstrip("/"),
            }
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
            "ONTO_PORTAL_API_KEY": self.settings.ontoportal_api_key,
        }

    def _command(
        self,
        *,
        prompt: str,
        workspace: Path,
        task: str = "edit",
        retrieved_context: str = "",
        citation_labels: list[str] | None = None,
        session_id: str | None = None,
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
        if session_id:
            command.extend(["--session", str(session_id)])
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
        if self._uses_antigravity_account_auth():
            return normalize_antigravity_model_ref(
                self.account_auth.model_ref if self.account_auth else None,
                default=self.settings.opencode_antigravity_model or self.settings.opencode_model or "",
            )
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

        api_mcp_name = self.settings.opencode_mcp_name
        rag_mcp_name = self.settings.opencode_rag_mcp_name
        search_guidance = (
            "- Antigravity auth is active: use the google_search tool for web/domain research when current external evidence is needed.\n"
            if self._uses_antigravity_account_auth()
            else "- Do not assume provider-native web search is available; if no explicit search tool is present, continue with RAG/API evidence and note the limitation.\n"
        )
        structured_guidance = (
            "- Ontology copilot schema mode is enabled: also write ontology-proposal.json, competency-questions.json, reuse-candidates.json, and validation-summary.json using schema_version ontology-copilot/v1.\n"
            "- In structured artifacts, include competency questions, reuse candidates, operations, evidence, assumptions, risks, and validation status; keep all changes proposal-only.\n"
            "- Prefer reuse or SKOS mappings before minting terms; require explicit human review before any owl:equivalentClass/owl:equivalentProperty claim.\n"
            if self.settings.ontology_copilot_enabled
            else ""
        )
        return (
            "You are preparing an ontology-edit proposal for MatPortal.\n"
            "Mandatory rules:\n"
            f"- Use the {rag_mcp_name} MCP server first for semantic retrieval, source chunks, and terminology discovery.\n"
            f"- Use the {api_mcp_name} MCP server for exact ontology/API state, classes, submissions, metadata, and full ontology inspection.\n"
            "- If the edit creates domain content or the user asks for ontology creation, research existing examples, standards, and terminology before modeling.\n"
            f"{search_guidance}"
            "- Write an edit plan before drafting artifacts, then inspect the ontology again after drafting or validation feedback.\n"
            "- Work only inside the current workspace.\n"
            f"- Use the `{_ONTOLOGY_TOOLKIT_DIR}/` templates and checklist as references for ontology artifacts and review notes.\n"
            "- Copy toolkit templates into new proposal files; do not edit toolkit files unless the user explicitly asks.\n"
            "- Do not commit, push, or access git remotes.\n"
            "- Write proposed artifacts, operator notes, provenance, and a draft submission package into files in this workspace.\n"
            f"{structured_guidance}"
            "- Prefer Turtle (.ttl) unless the user explicitly requests another format.\n"
            "- Finish with a concise operator-facing summary of proposed changes, validation status, sources used, and remaining assumptions.\n\n"
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
            if self.settings.opencode_block_dangerous_commands and tool == "bash":
                command = str((state.get("input") or {}).get("command") or "").strip()
                blocked_reason = self._blocked_bash_reason(command)
                if blocked_reason:
                    result.blocked = True
                    result.blocked_reason = blocked_reason
                    denial = f"Blocked OpenCode command: {blocked_reason}."
                    self._append_console_line(result, denial)
                    events.append({"type": "opencode_phase", "content": {"label": "Blocked unsafe command"}})
                    events.append({"type": "terminal_log", "content": {"line": denial}})
                    return events
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

    def _blocked_bash_reason(self, command: str) -> str:
        text = str(command or "").strip()
        if not text:
            return ""
        for pattern, reason in _BLOCKED_BASH_PATTERNS:
            if pattern.search(text):
                return reason
        return ""

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
        if self.account_auth:
            secrets.extend([self.account_auth.opencode_auth_json or "", self.account_auth.codex_auth_json or ""])
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
            robot_check = entry.get("robot")
            if isinstance(robot_check, dict) and robot_check.get("status") in {"skipped", "unavailable"}:
                warnings.append({"path": path_text, "message": str(robot_check.get("message") or "ROBOT validation unavailable.")})

        workflow_report, workflow_warnings, workflow_errors = self._workflow_completeness_findings(
            workspace=workspace,
            changed_files=changed_files,
        )
        warnings.extend(workflow_warnings)
        errors.extend(workflow_errors)

        passed = sum(1 for item in checked_files if item.get("status") == "passed")
        failed = sum(1 for item in checked_files if item.get("status") == "failed") + len(workflow_errors)
        skipped = sum(1 for item in checked_files if item.get("status") == "skipped")
        if failed:
            status_text = "failed"
        elif passed:
            status_text = "passed"
        else:
            status_text = "skipped"
        report = {
            "ok": failed == 0,
            "status": status_text,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "checked_files": checked_files,
            "errors": errors,
            "warnings": warnings,
            "workflow": workflow_report,
            "summary": f"{passed} passed, {failed} failed, {skipped} skipped",
        }
        return normalize_validation_report(report)

    def _workflow_completeness_findings(
        self,
        *,
        workspace: Path,
        changed_files: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
        present_paths = {
            str(item.get("path") or "").strip()
            for item in changed_files
            if str(item.get("status") or "").upper() != "D" and str(item.get("path") or "").strip()
        }
        try:
            root = workspace.resolve()
            for path in root.rglob("*"):
                if not path.is_file() or ".git" in path.parts:
                    continue
                try:
                    present_paths.add(path.resolve().relative_to(root).as_posix())
                except ValueError:
                    continue
        except OSError:
            pass

        present_names = {Path(path).name for path in present_paths}
        artifact_items = [
            {
                "name": artifact,
                "present": artifact in present_names,
                "path": next((path for path in sorted(present_paths) if Path(path).name == artifact), ""),
            }
            for artifact in _WORKFLOW_REQUIRED_ARTIFACTS
        ]
        missing = [item["name"] for item in artifact_items if not item["present"]]
        ontology_paths = sorted(path for path in present_paths if Path(path).suffix.lower() in _WORKFLOW_ONTOLOGY_SUFFIXES)
        structured_items = [
            {
                "name": artifact,
                "present": artifact in present_names,
                "path": next((path for path in sorted(present_paths) if Path(path).name == artifact), ""),
            }
            for artifact in sorted(STRUCTURED_ONTOLOGY_ARTIFACTS)
        ]
        structured_missing = [item["name"] for item in structured_items if not item["present"]]
        has_ontology_artifact = bool(ontology_paths)
        effective_missing = missing + (structured_missing if self.settings.ontology_copilot_enabled else [])
        workflow_missing = effective_missing + ([] if has_ontology_artifact else ["ontology-artifact"])
        workflow_ok = not effective_missing and has_ontology_artifact
        workflow_report: dict[str, Any] = {
            "strict": bool(self.settings.opencode_strict_workflow_enabled),
            "ontology_copilot": {
                "enabled": bool(self.settings.ontology_copilot_enabled),
                "ui_panels_enabled": bool(self.settings.ontology_ui_panels_enabled),
                "method_panel_enabled": bool(self.settings.ontology_method_panel_enabled),
                "reuse_enabled": bool(self.settings.ontology_reuse_enabled),
                "advanced_validation_enabled": bool(self.settings.ontology_advanced_validation_enabled),
                "reasoner_enabled": bool(self.settings.ontology_reasoner_enabled),
                "shacl_enabled": bool(self.settings.ontology_shacl_enabled),
                "build_profiles_enabled": bool(self.settings.ontology_build_profiles_enabled),
            },
            "required_artifacts": artifact_items,
            "structured_artifacts": structured_items,
            "ontology_artifact": {
                "present": has_ontology_artifact,
                "paths": ontology_paths,
                "suffixes": sorted(_WORKFLOW_ONTOLOGY_SUFFIXES),
            },
            "missing": workflow_missing,
            "ok": workflow_ok,
        }
        findings: list[dict[str, str]] = [
            {
                "path": artifact,
                "message": f"Workflow artifact is missing: {artifact}.",
            }
            for artifact in missing
        ]
        if self.settings.ontology_copilot_enabled:
            findings.extend(
                {
                    "path": artifact,
                    "message": f"Structured ontology-copilot artifact is missing: {artifact}.",
                }
                for artifact in structured_missing
            )
        if not has_ontology_artifact:
            findings.append(
                {
                    "path": "ontology-artifact",
                    "message": "Workflow artifact is missing: at least one ontology artifact (.ttl, .rdf, or .owl).",
                }
            )
        if not findings:
            return workflow_report, [], []
        if self.settings.opencode_strict_workflow_enabled:
            return workflow_report, [], findings
        return workflow_report, findings, []

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
                robot_check = self._validate_robot_file(path=path, display_path=display_path)
                if robot_check.get("status") == "failed":
                    return {
                        "path": display_path,
                        "kind": path.suffix.lstrip("."),
                        "status": "failed",
                        "parser": rdf_format,
                        "triples": len(graph),
                        "robot": robot_check,
                        "message": str(robot_check.get("message") or "ROBOT validation failed."),
                    }
                return {
                    "path": display_path,
                    "kind": path.suffix.lstrip("."),
                    "status": "passed",
                    "parser": rdf_format,
                    "triples": len(graph),
                    "robot": robot_check,
                }
            except Exception as exc:
                last_error = self._format_validation_error(exc, workspace=path.parent)
        return {
            "path": display_path,
            "kind": path.suffix.lstrip("."),
            "status": "failed",
            "message": last_error or "RDF parser rejected the artifact.",
        }

    def _validate_robot_file(self, *, path: Path, display_path: str) -> dict[str, Any]:
        command = self._robot_command("verify", path)
        if command is None:
            return {
                "status": "unavailable",
                "message": "ROBOT is not configured in this runtime.",
            }
        try:
            completed = subprocess.run(
                command,
                cwd=str(path.parent),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "tool": command[0],
                "command": self._redacted_command(command, workspace=path.parent),
                "message": "ROBOT verify timed out after 60 seconds.",
            }
        output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
        message = self._truncate_console_line(output, max_chars=400) if output else ""
        if completed.returncode == 0:
            result = {
                "status": "passed",
                "tool": command[0],
                "command": self._redacted_command(command, workspace=path.parent),
                "message": message or "ROBOT verify passed.",
            }
            report_result = self._build_robot_report(path=path)
            if report_result:
                result["report"] = report_result
            return result
        return {
            "status": "failed",
            "tool": command[0],
            "command": self._redacted_command(command, workspace=path.parent),
            "message": message or f"ROBOT verify failed for {display_path}.",
        }

    def _build_robot_report(self, *, path: Path) -> dict[str, Any] | None:
        output_path = path.with_name(f"{path.name}.robot-report.tsv")
        command = self._robot_command("report", path, output_path=output_path)
        if command is None:
            return None
        try:
            completed = subprocess.run(
                command,
                cwd=str(path.parent),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "path": output_path.name,
                "message": "ROBOT report timed out after 60 seconds.",
            }
        output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
        message = self._truncate_console_line(output, max_chars=400) if output else ""
        if completed.returncode == 0:
            return {
                "status": "passed",
                "path": output_path.name,
                "message": message or "ROBOT report generated.",
            }
        return {
            "status": "failed",
            "path": output_path.name,
            "message": message or "ROBOT report failed.",
        }

    def _robot_command(self, action: str, path: Path, *, output_path: Path | None = None) -> list[str] | None:
        if not bool(self.settings.opencode_robot_enabled):
            return None
        action = str(action or "").strip()
        if action not in {"verify", "report"}:
            return None
        args = [action, "--input", str(path)]
        if output_path is not None:
            args.extend(["--output", str(output_path)])
        robot_jar = self.settings.opencode_robot_jar_path
        if robot_jar and robot_jar.exists():
            return [
                self.settings.opencode_robot_java_path,
                "-jar",
                str(robot_jar),
                *args,
            ]
        robot_path = shutil.which("robot")
        if robot_path:
            return [robot_path, *args]
        return None

    def _redacted_command(self, command: list[str], *, workspace: Path) -> list[str]:
        workspace_text = str(workspace)
        return [str(item).replace(workspace_text, "<workspace>") for item in command]

    def _validate_json_file(self, *, path: Path, display_path: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entry: dict[str, Any] = {"path": display_path, "kind": "json", "status": "passed", "parser": "json"}
            if self.settings.ontology_copilot_enabled and path.name in STRUCTURED_ONTOLOGY_ARTIFACTS:
                schema_check = validate_ontology_proposal_payload(payload, artifact_name=path.name)
                entry["schema"] = schema_check
                if schema_check.get("status") == "failed":
                    entry["status"] = "failed"
                    entry["message"] = str(schema_check.get("message") or "Ontology proposal schema validation failed.")
            return entry
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
