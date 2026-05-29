"""Ontology-copilot proposal and validation helpers."""

from .diagnostics import Diagnostic, normalize_validation_report
from .proposals import OntologyProposal, validate_ontology_proposal_payload

__all__ = [
    "Diagnostic",
    "OntologyProposal",
    "normalize_validation_report",
    "validate_ontology_proposal_payload",
]
