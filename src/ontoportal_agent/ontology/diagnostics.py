from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DiagnosticStatus = Literal["passed", "warning", "failed", "blocked", "skipped", "unavailable"]
DiagnosticSeverity = Literal["info", "warning", "error", "blocked"]


_STATUS_ALIASES: dict[str, DiagnosticStatus] = {
    "ok": "passed",
    "pass": "passed",
    "passed": "passed",
    "success": "passed",
    "warn": "warning",
    "warning": "warning",
    "warnings": "warning",
    "error": "failed",
    "errors": "failed",
    "fail": "failed",
    "failed": "failed",
    "blocked": "blocked",
    "skip": "skipped",
    "skipped": "skipped",
    "unavailable": "unavailable",
}


def normalize_status(value: Any, *, default: DiagnosticStatus = "skipped") -> DiagnosticStatus:
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return _STATUS_ALIASES.get(normalized, default)


def severity_for_status(status: DiagnosticStatus) -> DiagnosticSeverity:
    if status == "failed":
        return "error"
    if status == "blocked":
        return "blocked"
    if status in {"warning", "unavailable"}:
        return "warning"
    return "info"


class Diagnostic(BaseModel):
    """User-facing normalized ontology validation diagnostic."""

    status: DiagnosticStatus
    severity: DiagnosticSeverity | None = None
    source: str = "validation"
    path: str = ""
    message: str = ""
    code: str = ""
    entity_iri: str = ""
    evidence: list[str] = Field(default_factory=list)
    suggestion: str = ""

    def model_post_init(self, __context: Any) -> None:
        if self.severity is None:
            self.severity = severity_for_status(self.status)


def _diagnostic_from_item(
    item: dict[str, Any],
    *,
    source: str,
    default_status: DiagnosticStatus,
    default_message: str,
) -> Diagnostic:
    status = normalize_status(item.get("status"), default=default_status)
    return Diagnostic(
        status=status,
        severity=severity_for_status(status),
        source=str(item.get("source") or source),
        path=str(item.get("path") or ""),
        message=str(item.get("message") or default_message),
        code=str(item.get("code") or item.get("kind") or ""),
        entity_iri=str(item.get("entity_iri") or item.get("entity") or ""),
        evidence=[str(value) for value in item.get("evidence", []) if str(value or "").strip()]
        if isinstance(item.get("evidence"), list)
        else [],
        suggestion=str(item.get("suggestion") or ""),
    )


def normalize_validation_report(report: dict[str, Any]) -> dict[str, Any]:
    """Add normalized diagnostics without removing legacy validation report fields."""

    normalized = dict(report or {})
    diagnostics: list[Diagnostic] = []

    for item in normalized.get("checked_files", []) or []:
        if not isinstance(item, dict):
            continue
        diagnostics.append(
            _diagnostic_from_item(
                item,
                source="artifact-validator",
                default_status="skipped",
                default_message="Artifact validation completed.",
            )
        )
        robot = item.get("robot")
        if isinstance(robot, dict):
            diagnostics.append(
                _diagnostic_from_item(
                    {**robot, "path": robot.get("path") or item.get("path") or ""},
                    source="robot",
                    default_status="unavailable",
                    default_message="ROBOT validation did not complete.",
                )
            )

    for item in normalized.get("warnings", []) or []:
        if isinstance(item, dict):
            diagnostics.append(
                _diagnostic_from_item(
                    item,
                    source="workflow",
                    default_status="warning",
                    default_message="Workflow warning.",
                )
            )

    for item in normalized.get("errors", []) or []:
        if isinstance(item, dict):
            diagnostics.append(
                _diagnostic_from_item(
                    item,
                    source="workflow",
                    default_status="failed",
                    default_message="Workflow error.",
                )
            )

    workflow = normalized.get("workflow")
    if isinstance(workflow, dict):
        status = "passed" if workflow.get("ok") else "warning"
        if workflow.get("strict") and not workflow.get("ok"):
            status = "failed"
        diagnostics.append(
            Diagnostic(
                status=status,  # type: ignore[arg-type]
                severity=severity_for_status(status),  # type: ignore[arg-type]
                source="workflow",
                path="workflow",
                message="Workflow completeness check passed." if workflow.get("ok") else "Workflow completeness has missing artifacts.",
                code="workflow_completeness",
            )
        )

    normalized["diagnostics"] = [item.model_dump() for item in diagnostics]
    normalized["diagnostic_summary"] = {
        "total": len(diagnostics),
        "failed": sum(1 for item in diagnostics if item.status == "failed"),
        "blocked": sum(1 for item in diagnostics if item.status == "blocked"),
        "warnings": sum(1 for item in diagnostics if item.status in {"warning", "unavailable"}),
        "passed": sum(1 for item in diagnostics if item.status == "passed"),
        "skipped": sum(1 for item in diagnostics if item.status == "skipped"),
    }
    return normalized
