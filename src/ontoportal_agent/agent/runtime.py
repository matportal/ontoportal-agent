from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage

from ..config import get_settings
from ..ontology_repository import OntologyRepository
from .graph import build_agent_graph
from .state import AgentState


class OntoPortalAgent:
    def __init__(self, *, repository: Optional[OntologyRepository] = None):
        self.settings = get_settings()
        self.repository = repository or OntologyRepository()
        self.graph = build_agent_graph(self.repository).compile()

    def invoke(self, question: str) -> str:
        state: AgentState = {"user_input": question}
        final_state = self.graph.invoke(state)
        return final_state.get("final_response", "No response generated.")

    def stream(self, question: str):
        state: AgentState = {"user_input": question}
        yield from self.graph.stream(state)
