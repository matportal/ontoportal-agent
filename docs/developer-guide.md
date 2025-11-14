# OntoPortal Agent Developer Guide

This guide summarizes the project layout, extension points, and testing strategy for contributors.

## Project Layout
```
ontoportal-agent/
├── pyproject.toml
├── src/ontoportal_agent/
│   ├── agent/            # LangGraph wiring & state definitions
│   ├── cli.py            # Typer-based CLI entry points
│   ├── config.py         # Pydantic settings loader
│   ├── ontology_repository.py
│   ├── sandbox.py
│   ├── publishing.py
│   └── rag_client.py
└── tests/
```

Key directories:
- `src/ontoportal_agent/agent` - graph construction (`graph.py`), runtime wrapper (`runtime.py`), and
  state definition (`state.py`).
- `src/ontoportal_agent/mcp_client.py` - lightweight client for Model Context Protocol endpoints.
- `tests/test_tensile_ontology.py` - end-to-end example that stubs LLMs, ROBOT, and the MatPortal API
  to validate planning, sandbox execution, and publishing.

## Extending the LangGraph
1. Review `agent/graph.py` to understand the current state machine.
2. Add new nodes with `graph.add_node("name", handler)` and connect them via `graph.add_edge` or
   `graph.add_conditional_edges`.
3. Update `AgentState` (`agent/state.py`) if additional fields must be passed between steps.
4. Keep node handlers pure functions where possible; this simplifies testing and streaming.

## Custom Tools & Sandboxed Actions
- Wrap reusable actions as `langchain_core.tools.StructuredTool` instances in `agent/tools.py`.
- When writing sandbox code, ensure the snippet finishes by calling
  `ontology_repo.save_graph(graph, workspace, filename)` to persist changes.
- Prefer deterministic scripts (ROBOT, SHACL, SPARQL) so reviewers can reproduce the output locally.

## Configuration & MCP Integration
- Runtime settings are provided through environment variables prefixed with `ONTOAGENT_`.
- If `ONTOAGENT_MCP_ENDPOINTS` is not set, the agent automatically points to the OntoPortal RAG MCP
  adapter at `<ONTOAGENT_RAG_BASE_URL>/mcp`.
- Register additional MCP endpoints to expose auxiliary tools (e.g. metadata enrichment, validation
  services) without modifying the agent code.

## Testing
Install the test dependencies and run the suite:

```bash
pip install -e .[test]
pytest
```

- `tests/test_config.py` ensures settings parsing behaves as expected.
- `tests/test_tensile_ontology.py` demonstrates how to monkeypatch LLMs, subprocess calls, and HTTP
  clients to cover the full plan-execute-publish flow.

Integrate these tests into CI to catch regressions whenever dependencies or graph logic change.
