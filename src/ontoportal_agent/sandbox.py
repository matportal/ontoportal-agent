from __future__ import annotations

import io
import textwrap
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict

from rdflib import Graph
import owlready2  # noqa: F401 - imported for sandbox convenience

from .ontology_repository import OntologyRepository, OntologyArtifact


@dataclass
class SandboxResult:
    stdout: str
    locals: Dict[str, Any]


class PythonSandbox:
    """
    Executes user-provided Python snippets with a curated set of helpers.
    The sandbox is intentionally lightweight and should be treated as
    semi-trusted; callers are responsible for validating user input.
    """

    def __init__(self, repository: OntologyRepository):
        self.repository = repository

    def run(
        self,
        code: str,
        *,
        graph: Graph | None = None,
        artifact: OntologyArtifact | None = None,
        extra_globals: dict[str, Any] | None = None,
    ) -> SandboxResult:
        prepared = textwrap.dedent(code)
        global_scope: Dict[str, Any] = {
            "Graph": Graph,
            "ontology_repo": self.repository,
            "artifact": artifact,
            "graph": graph,
        }
        if extra_globals:
            global_scope.update(extra_globals)

        local_scope: Dict[str, Any] = {}
        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            exec(prepared, global_scope, local_scope)
        return SandboxResult(stdout=stdout_buffer.getvalue(), locals=local_scope)
