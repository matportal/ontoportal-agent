from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = "ontology-copilot/v1"
STRUCTURED_ONTOLOGY_ARTIFACTS = {
    "ontology-proposal.json",
    "competency-questions.json",
    "reuse-candidates.json",
    "validation-summary.json",
}

OperationKind = Literal[
    "create_class",
    "create_property",
    "add_annotation",
    "add_axiom",
    "add_mapping",
    "deprecate_term",
    "rename_label",
    "move_subclass",
]
EntityType = Literal["class", "object_property", "data_property", "annotation_property", "individual", "ontology", "mapping"]
MappingRelation = Literal["skos:exactMatch", "skos:closeMatch", "skos:broadMatch", "skos:narrowMatch", "skos:relatedMatch"]
ValidationStatus = Literal["passed", "warning", "failed", "blocked", "skipped", "unavailable"]

_LOCAL_PATH_RE = re.compile(r"(^|\s)(/home/|/tmp/|/var/|/etc/|[A-Za-z]:\\)")
_SECRET_LIKE_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer\s+|client[_-]?secret|password|token)")


def _reject_sensitive_text(value: str) -> str:
    text = str(value or "")
    if _LOCAL_PATH_RE.search(text):
        raise ValueError("must not contain absolute local filesystem paths")
    if _SECRET_LIKE_RE.search(text):
        raise ValueError("must not contain secret-like text")
    return text


def _scan_sensitive_payload(value: Any, *, path: str = "$") -> str | None:
    if isinstance(value, str):
        try:
            _reject_sensitive_text(value)
        except ValueError as exc:
            return f"{path}: {exc}"
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            try:
                _reject_sensitive_text(key_text)
            except ValueError as exc:
                return f"{path}.{key_text}: {exc}"
            finding = _scan_sensitive_payload(item, path=f"{path}.{key_text}")
            if finding:
                return finding
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            finding = _scan_sensitive_payload(item, path=f"{path}.{index}")
            if finding:
                return finding
        return None
    return None


class StrictProposalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceRef(StrictProposalModel):
    source: str = Field(min_length=1, max_length=200)
    citation: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=500)
    quote: str = Field(default="", max_length=1000)

    @field_validator("source", "citation", "url", "quote")
    @classmethod
    def _safe_text(cls, value: str) -> str:
        return _reject_sensitive_text(value)


class CompetencyQuestion(StrictProposalModel):
    id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=500)
    expected_answer: str = Field(default="", max_length=1000)
    status: Literal["draft", "answered", "blocked"] = "draft"

    @field_validator("id", "question", "expected_answer")
    @classmethod
    def _safe_text(cls, value: str) -> str:
        return _reject_sensitive_text(value)


class ReuseCandidate(StrictProposalModel):
    label: str = Field(min_length=1, max_length=200)
    iri: str = Field(default="", max_length=500)
    source_ontology: str = Field(default="", max_length=120)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_action: Literal["reuse", "map", "extend", "reject", "needs_review"] = "needs_review"
    rationale: str = Field(default="", max_length=1000)

    @field_validator("label", "iri", "source_ontology", "rationale")
    @classmethod
    def _safe_text(cls, value: str) -> str:
        return _reject_sensitive_text(value)


class ProposalOperation(StrictProposalModel):
    operation: OperationKind
    entity_type: EntityType
    iri: str = Field(default="", max_length=500)
    label: str = Field(default="", max_length=200)
    parent_iri: str = Field(default="", max_length=500)
    target_iri: str = Field(default="", max_length=500)
    mapping_relation: MappingRelation | None = None
    turtle: str = Field(default="", max_length=8000)
    rationale: str = Field(min_length=1, max_length=2000)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @field_validator("iri", "label", "parent_iri", "target_iri", "turtle", "rationale")
    @classmethod
    def _safe_text(cls, value: str) -> str:
        return _reject_sensitive_text(value)

    @model_validator(mode="after")
    def _mapping_relation_for_mapping(self) -> "ProposalOperation":
        if self.operation == "add_mapping" and self.mapping_relation is None:
            raise ValueError("add_mapping operations must use a SKOS mapping relation")
        if self.operation.startswith("create_") and not self.label:
            raise ValueError("create operations must include a human-readable label")
        if not (self.iri or self.target_iri or self.turtle):
            raise ValueError("operation must identify an IRI, target IRI, or Turtle snippet")
        return self


class OntologyProposal(StrictProposalModel):
    schema_version: Literal[SCHEMA_VERSION]
    proposal_id: str = Field(default="", max_length=120)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=2000)
    ontology_acronym: str = Field(default="", max_length=80)
    goals: list[str] = Field(default_factory=list, max_length=20)
    scope: str = Field(default="", max_length=2000)
    competency_questions: list[CompetencyQuestion] = Field(default_factory=list)
    reuse_candidates: list[ReuseCandidate] = Field(default_factory=list)
    operations: list[ProposalOperation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    risks: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("proposal_id", "title", "summary", "ontology_acronym", "scope")
    @classmethod
    def _safe_text(cls, value: str) -> str:
        return _reject_sensitive_text(value)

    @field_validator("goals", "assumptions", "risks")
    @classmethod
    def _safe_text_list(cls, value: list[str]) -> list[str]:
        return [_reject_sensitive_text(item) for item in value]

    @model_validator(mode="after")
    def _requires_methodology_content(self) -> "OntologyProposal":
        if not self.competency_questions:
            raise ValueError("at least one competency question is required")
        if not self.operations:
            raise ValueError("at least one proposal operation is required")
        return self


class CompetencyQuestionsArtifact(StrictProposalModel):
    schema_version: Literal[SCHEMA_VERSION]
    questions: list[CompetencyQuestion] = Field(min_length=1)


class ReuseCandidatesArtifact(StrictProposalModel):
    schema_version: Literal[SCHEMA_VERSION]
    candidates: list[ReuseCandidate] = Field(min_length=1)


class ValidationSummaryArtifact(StrictProposalModel):
    schema_version: Literal[SCHEMA_VERSION]
    status: ValidationStatus
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="", max_length=2000)

    @field_validator("summary")
    @classmethod
    def _safe_summary(cls, value: str) -> str:
        return _reject_sensitive_text(value)


def _model_for_artifact(artifact_name: str) -> type[BaseModel] | None:
    name = artifact_name.strip().split("/")[-1]
    if name == "ontology-proposal.json":
        return OntologyProposal
    if name == "competency-questions.json":
        return CompetencyQuestionsArtifact
    if name == "reuse-candidates.json":
        return ReuseCandidatesArtifact
    if name == "validation-summary.json":
        return ValidationSummaryArtifact
    return None


def validate_ontology_proposal_payload(payload: Any, *, artifact_name: str) -> dict[str, Any]:
    """Validate known proposal artifacts and return a legacy-compatible validation entry."""

    model = _model_for_artifact(artifact_name)
    if model is None:
        return {"schema": "", "status": "skipped", "message": "No ontology proposal schema registered for this JSON artifact."}
    sensitive_finding = _scan_sensitive_payload(payload)
    if sensitive_finding:
        return {
            "schema": model.__name__,
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "message": sensitive_finding,
        }
    try:
        parsed = model.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(part) for part in first.get("loc", [])) or artifact_name
        msg = str(first.get("msg") or "Schema validation failed")
        return {
            "schema": model.__name__,
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "message": f"{loc}: {msg}",
        }
    return {
        "schema": model.__name__,
        "schema_version": getattr(parsed, "schema_version", SCHEMA_VERSION),
        "status": "passed",
    }
