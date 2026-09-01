import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdflib import Graph, Namespace, RDF, RDFS
from langchain_core.messages import AIMessage

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "langgraph.graph" not in sys.modules:
    END_SENTINEL = object()

    class DummyStateGraph:
        def __init__(self, *_args, **_kwargs):
            self.nodes = {}
            self.entry_point = None
            self.edges = {}
            self.conditional_edges = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def set_entry_point(self, name):
            self.entry_point = name

        def add_edge(self, source, target):
            self.edges.setdefault(source, []).append(target)

        def add_conditional_edges(self, source, router, mapping):
            self.conditional_edges[source] = (router, mapping)

        def compile(self):
            graph = self

            class _ExecutableGraph:
                def __init__(self, state_graph):
                    self._graph = state_graph

                def _call_router(self, router, state):
                    if hasattr(router, "invoke"):
                        return router.invoke(state)
                    return router(state)

                def _step(self, node_name, state):
                    func = self._graph.nodes[node_name]
                    return func(state)

                def invoke(self, state):
                    current = self._graph.entry_point
                    state = dict(state)
                    safety = 0
                    while True:
                        state = self._step(current, state)
                        router_info = self._graph.conditional_edges.get(current)
                        if router_info:
                            router, mapping = router_info
                            key = self._call_router(router, state)
                            if key not in mapping:
                                raise KeyError(f"No mapping for key '{key}' in conditional edges of '{current}'")
                            next_node = mapping[key]
                        else:
                            outgoing = self._graph.edges.get(current, [])
                            next_node = outgoing[0] if outgoing else END_SENTINEL

                        if next_node is END_SENTINEL:
                            return state

                        current = next_node
                        safety += 1
                        if safety > 100:
                            raise RuntimeError("StateGraph execution exceeded safety limit.")

                def stream(self, state):
                    yield self.invoke(state)

            return _ExecutableGraph(graph)

    langgraph_module = types.ModuleType("langgraph")
    langgraph_graph_module = types.ModuleType("langgraph.graph")
    langgraph_graph_module.StateGraph = DummyStateGraph
    langgraph_graph_module.END = END_SENTINEL
    langgraph_module.graph = langgraph_graph_module
    langgraph_module.__spec__ = importlib.machinery.ModuleSpec("langgraph", loader=None)
    langgraph_graph_module.__spec__ = importlib.machinery.ModuleSpec("langgraph.graph", loader=None)
    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = langgraph_graph_module

if "owlready2" not in sys.modules:
    owlready_module = types.ModuleType("owlready2")
    owlready_module.__spec__ = importlib.machinery.ModuleSpec("owlready2", loader=None)
    sys.modules["owlready2"] = owlready_module

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.config import AgentSettings


class StubChatOpenAI:
    def __init__(self, *_, **__):
        polymer_ns = "http://example.org/polymer/"
        self.plan = {
            "workspace": "polymer-tensile",
            "actions": [
                {
                    "description": "Create a polymer tensile test ontology using rdflib.",
                    "artifact": "tensile_test.ttl",
                    "create": True,
                    "format": "turtle",
                    "code": f"""
from rdflib import Namespace, RDF, RDFS, Literal
EX = Namespace("{polymer_ns}")
graph.bind("ex", EX)
graph.add((EX.PolymerTensileTest, RDF.type, RDFS.Class))
graph.add((EX.PolymerSpecimen, RDF.type, RDFS.Class))
graph.add((EX.PolymerSpecimen, RDFS.label, Literal("Polymer specimen for tensile testing")))
graph.add((EX.hasSpecimen, RDF.type, RDF.Property))
graph.add((EX.PolymerTensileTest, EX.hasSpecimen, EX.PolymerSpecimen))
ontology_repo.save_graph(graph, workspace, "tensile_test.ttl")
print("Ontology created")
""",
                },
                {
                    "description": "Validate the ontology with ROBOT verify.",
                    "artifact": "tensile_test.ttl",
                    "create": False,
                    "format": "turtle",
                    "code": """
import subprocess
command = ["robot", "verify", "--input", str(artifact.path)]
result = subprocess.run(command, check=True, capture_output=True, text=True)
print(result.stdout or "ROBOT verification passed.")
""",
                },
            ],
            "publish": {
                "acronym": "POLYTENS",
                "artifact": "tensile_test.ttl",
                "contact_email": "polymer@example.org",
                "notes": "Private tensile test ontology",
                "private": True,
            },
        }

    def _extract_messages(self, value):
        if hasattr(value, "to_messages"):
            return value.to_messages()
        if isinstance(value, dict) and "messages" in value:
            return value["messages"]
        return value

    def __call__(self, value, **kwargs):
        return self.invoke(value, **kwargs)

    def invoke(self, value, **_kwargs):
        messages = self._extract_messages(value)
        if not messages:
            return AIMessage(content="OK")

        system_content = messages[0].content
        if "Classify the user's intent" in system_content:
            return AIMessage(content="EDIT")
        if "senior ontology engineer" in system_content:
            return AIMessage(content=json.dumps(self.plan))
        if "You are the OntoPortal assistant." in system_content:
            return AIMessage(content="Ontology update submitted to OntoPortal.")
        return AIMessage(content="OK")


class StubRagClient:
    def query(self, _question):
        return SimpleNamespace(answer="No additional context required.", sources=[])


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_agent_creates_validates_and_submits_private_polymer_tensile_ontology(monkeypatch, tmp_path):
    from ontoportal_agent import config as config_module
    from ontoportal_agent.agent import graph as graph_module
    from ontoportal_agent import publishing as publishing_module
    from ontoportal_agent import ontology_repository as repository_module
    from ontoportal_agent import rag_client as rag_module
    from ontoportal_agent.agent import runtime as runtime_module
    from ontoportal_agent.agent.runtime import OntoPortalAgent

    # Configure agent settings for automated publishing during the test.
    config_module.get_settings.cache_clear()
    settings = AgentSettings(
        OPENAI_API_KEY="test",
        ONTOPORTAL_API_KEY="mat-key",
        ONTOLOGY_WORKDIR=str(tmp_path),
        REQUIRE_MANUAL_APPROVAL=False,
        _env_file=None,
    )
    monkeypatch.setattr(config_module, "get_settings", lambda: settings, raising=False)
    monkeypatch.setattr(graph_module, "get_settings", lambda: settings, raising=False)
    monkeypatch.setattr(publishing_module, "get_settings", lambda: settings, raising=False)
    monkeypatch.setattr(repository_module, "get_settings", lambda: settings, raising=False)
    monkeypatch.setattr(rag_module, "get_settings", lambda: settings, raising=False)
    monkeypatch.setattr(runtime_module, "get_settings", lambda: settings, raising=False)

    # Stub external dependencies: LLM, RAG, ROBOT CLI, and MatPortal REST API.
    monkeypatch.setattr(graph_module, "ChatOpenAI", StubChatOpenAI, raising=False)
    monkeypatch.setattr(graph_module, "RagClient", lambda: StubRagClient(), raising=False)

    robot_calls = []

    def fake_run(cmd, check=False, capture_output=False, text=False):
        robot_calls.append({"cmd": cmd, "check": check, "capture_output": capture_output, "text": text})
        return SimpleNamespace(returncode=0, stdout="ROBOT verification passed.", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run, raising=False)

    submissions = []

    def fake_post(url, headers=None, data=None, timeout=None):
        submissions.append({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return DummyResponse({"submissionId": "POLYTENS-001"})

    monkeypatch.setattr(publishing_module.requests, "post", fake_post)

    agent = OntoPortalAgent()

    request = "Generate a tensile test ontology for polymers, validate it, and submit privately to MatPortal."
    response = agent.invoke(request)

    workspace_dir = tmp_path / "polymer-tensile"
    artifact_path = workspace_dir / "tensile_test.ttl"

    assert artifact_path.exists(), "Ontology artifact was not created."

    graph = Graph()
    graph.parse(artifact_path, format="turtle")
    ex = Namespace("http://example.org/polymer/")

    assert (ex.PolymerTensileTest, RDF.type, RDFS.Class) in graph, "Expected class definition missing."
    assert (ex.PolymerTensileTest, ex.hasSpecimen, ex.PolymerSpecimen) in graph, "Expected tensile test relation missing."

    assert robot_calls, "ROBOT verification was not invoked."
    robot_command = robot_calls[0]["cmd"]
    assert robot_command[:3] == ["robot", "verify", "--input"], "Unexpected ROBOT validation command."

    assert submissions, "No MatPortal submission was performed."
    payload = json.loads(submissions[0]["data"])
    assert payload.get("isPrivate") is True, "Submission was not marked as private."
    assert payload["filename"] == "tensile_test.ttl"

    assert "Ontology update submitted to OntoPortal." in response
