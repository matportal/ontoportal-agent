from pathlib import Path

from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL

from ontoportal_agent import config as config_module
from ontoportal_agent.ontology_repository import OntologyArtifact, OntologyRepository


def _example_graph() -> Graph:
    graph = Graph()
    ns = Namespace("http://example.org/material-science#")
    graph.add((ns.TensileStrengthMeasurement, RDF.type, OWL.Class))
    graph.add((ns.TensileStrengthMeasurement, RDFS.label, Literal("Tensile Strength Measurement")))
    return graph


def _configure_repository_env(monkeypatch) -> None:
    monkeypatch.setenv("ONTOAGENT_OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ONTOAGENT_ONTOPORTAL_API_KEY", "test-ontoportal-key")
    config_module.get_settings.cache_clear()


def test_save_graph_accepts_artifact_object(tmp_path: Path, monkeypatch):
    _configure_repository_env(monkeypatch)
    repository = OntologyRepository(workdir=tmp_path)
    workspace = repository.create_workspace("session")
    artifact = OntologyArtifact(path=workspace / "mso.ttl", format="ttl")

    saved = repository.save_graph(_example_graph(), workspace, artifact)

    assert saved.path == artifact.path
    assert saved.format == "ttl"
    assert saved.path.exists()


def test_save_graph_accepts_relative_path(tmp_path: Path, monkeypatch):
    _configure_repository_env(monkeypatch)
    repository = OntologyRepository(workdir=tmp_path)
    workspace = repository.create_workspace("session")

    saved = repository.save_graph(_example_graph(), workspace, Path("nested") / "mso.ttl")

    assert saved.path == workspace / "nested" / "mso.ttl"
    assert saved.path.exists()
