from __future__ import annotations

from typing import Any, Dict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..ontology_repository import OntologyRepository, OntologyArtifact
from ..sandbox import PythonSandbox


class SandboxInput(BaseModel):
    code: str = Field(..., description="Python code to execute in the sandbox")
    artifact_path: str = Field(..., description="Path to the ontology artifact inside the workspace")


def build_sandbox_tool(repository: OntologyRepository) -> StructuredTool:
    sandbox = PythonSandbox(repository)

    def _invoke(code: str, artifact_path: str) -> Dict[str, Any]:
        artifact = OntologyArtifact(path=repository.workdir / artifact_path)
        graph = repository.load_graph(artifact)
        result = sandbox.run(code, graph=graph, artifact=artifact)
        return {
            "stdout": result.stdout,
            "locals": {k: str(v) for k, v in result.locals.items()},
        }

    return StructuredTool.from_function(
        func=_invoke,
        name="python_sandbox",
        description="Execute trusted Python code against an ontology artifact.",
        args_schema=SandboxInput,
    )
