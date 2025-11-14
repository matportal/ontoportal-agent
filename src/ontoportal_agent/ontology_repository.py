from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from rdflib import Graph

from .config import get_settings


@dataclass
class OntologyArtifact:
    path: Path
    format: str = "ttl"

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def write_text(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")


class OntologyRepository:
    """Manages ontology documents prepared by the agent."""

    def __init__(self, *, workdir: Optional[Path] = None):
        settings = get_settings()
        self.workdir = workdir or settings.ontology_workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

    def create_workspace(self, name: str) -> Path:
        workspace = self.workdir / name
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def load_graph(self, artifact: OntologyArtifact) -> Graph:
        graph = Graph()
        graph.parse(str(artifact.path), format=artifact.format)
        return graph

    def save_graph(self, graph: Graph, workspace: Path, filename: str, format: str = "turtle") -> OntologyArtifact:
        outfile = workspace / filename
        graph.serialize(destination=str(outfile), format=format)
        return OntologyArtifact(path=outfile, format="ttl" if format == "turtle" else format)

    def list_artifacts(self, workspace: Path) -> Iterable[OntologyArtifact]:
        for file in workspace.iterdir():
            if file.suffix in {".ttl", ".rdf", ".owl"}:
                yield OntologyArtifact(path=file, format=file.suffix.lstrip("."))

    def export_metadata(self, workspace: Path) -> str:
        manifest = {
            "artifacts": [art.path.name for art in self.list_artifacts(workspace)],
            "workspace": str(workspace),
        }
        return json.dumps(manifest, indent=2)
