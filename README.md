# OntoPortal Agent

`ontoportal-agent` is the automation layer that bridges the OntoPortal thesaurus ecosystem with
LLM-assisted authoring, validation, and publication workflows. It combines LangChain, LangGraph,
and a controlled Python sandbox to help ontology engineers discover knowledge, stage edits, and
submit changes back to MatPortal / OntoPortal with explicit human approval.

Additional documentation lives under `docs/`:
- `docs/index.md` - navigation hub
- `docs/user-guide.md` - installation and day-to-day usage
- `docs/architecture.md` and `docs/developer-guide.md` - internals and extension points

## Highlights
- **Retrieval Augmented Generation (RAG)** - Delegates fact-finding to the local `ontoportal-rag-mcp`
  FastAPI service and returns fully cited answers.
- **Guided editing plans** - Uses LangGraph to classify intents, script ontology mutations, and
  capture change notes before publication.
- **Python sandbox** - Executes rdflib/owlready2 snippets inside a temporary workspace so edits
  are auditable and reversible.
- **Publication pipeline** - Wraps the OntoPortal REST API to upload vetted ontology artifacts
  once a reviewer confirms the changes.
- **Model Context Protocol (MCP)** - Discovers additional tools from MCP endpoints and makes
  them available to the agent without code changes.

---

## Architecture Overview

```
+-------------+        +------------------------------------------+
|  CLI / API  | -----> | LangGraph state machine (agent/graph.py) |
+-------------+        +------------------------------------------+
                            |                      |
                            |                      |
                       RETRIEVE path          EDIT path
                            |                      |
                            v                      v
                   +----------------+     +----------------------+
                   |  RAGClient     |     |  PythonSandbox       |
                   |  -> FastAPI    |     |  + OntologyRepository|
                   +----------------+     +----------------------+
                                                   |
                                                   v
                                         +----------------------+
                                         | OntoPortalPublisher  |
                                         | -> OntoPortal REST   |
                                         +----------------------+
```

### Request lifecycle
1. **Intent classification** - An OpenAI model labels the input as `RETRIEVE` or `EDIT`.
2. **Retrieval path** - The agent queries the OntoPortal-RAG service, gathers citations, and
   composes a natural-language answer for the user.
3. **Edit path** - The LLM produces a JSON plan that lists sandbox actions, artifacts to touch,
   and optional publication metadata.
4. **Sandbox execution** - Each action runs inside a temporary workspace backed by the
   `OntologyRepository`, ensuring files land under `<ONTOLOGY_WORKDIR>/<workspace>/`.
5. **Approval gate** - The agent summarizes the proposed edits and pauses. With manual approval
   disabled (`REQUIRE_MANUAL_APPROVAL=false`), publishing can proceed automatically.
6. **Publishing** - `OntoPortalPublisher` encodes the artifact and contacts the OntoPortal REST
   API to create a new submission.

Core modules:

| Module | Responsibility |
| ------ | -------------- |
| `agent/graph.py` | Builds the LangGraph state machine; orchestrates routing, planning, execution, and publication. |
| `agent/runtime.py` | User-facing entry point (`OntoPortalAgent`) and streaming helpers. |
| `config.py` | Loads environment-driven settings via Pydantic. |
| `rag_client.py` | HTTP client for the OntoPortal RAG service. |
| `ontology_repository.py` | Manages workspaces, rdflib serialization, and artifact metadata. |
| `sandbox.py` | Runs trusted Python code with controlled globals for ontology editing. |
| `publishing.py` | Integrates with OntoPortal/MatPortal REST endpoints. |
| `mcp_client.py` | Discovers and invokes tools exposed through MCP endpoints. |

---

## Getting Started (Users)

### Prerequisites
- Python >= 3.11
- Access to a running OntoPortal RAG FastAPI service (see the separate `ontoportal-rag-mcp` project)
- OntoPortal REST API key with submission privileges
- OpenAI-compatible chat completion model (default: `gpt-4o-mini`)

### Installation
```bash
cd /home/todor/ontoportal/ontoportal-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure environment
Create `.env` in the project root and populate the required values:

```dotenv
ONTOAGENT_OPENAI_API_KEY=sk-...
ONTOAGENT_ONTOPORTAL_API_KEY=matportal-key
ONTOAGENT_RAG_BASE_URL=http://localhost:8000
ONTOAGENT_RAG_QUERY_PATH=/api/v1/query
ONTOAGENT_ONTOLOGY_WORKDIR=/tmp/ontoportal-agent
ONTOAGENT_MCP_API_KEY=change-me
ONTOAGENT_MCP_RAG_TOOL_NAME=rag_query
# Required before exposing /api/v1/me/* or either chat stream
ONTOAGENT_INTERNAL_API_TOKEN=replace-me
ONTOAGENT_USER_CONTEXT_SECRET=replace-me
ONTOAGENT_ENCRYPTION_KEY_CURRENT=replace-with-a-valid-32-byte-key
# Optional overrides
ONTOAGENT_OPENAI_API_BASE=https://api.openai.com/v1
ONTOAGENT_LLM_MODEL=gpt-4o-mini
ONTOAGENT_REQUIRE_MANUAL_APPROVAL=true
ONTOAGENT_MCP_ENDPOINTS=http://localhost:8000/mcp
```

Environment keys are documented in `src/ontoportal_agent/config.py`. They accept either the
`ONTOAGENT_*` variant (preferred) or the alias shown above.

### Launch the chat interface
```bash
python -m ontoportal_agent.cli chat
```

### Launch the streaming API for the web UI
```bash
python -m ontoportal_agent.server
```

The endpoint used by the Rails assistant UI is:
- `POST /api/v1/chat/stream` (Server-Sent Events response)

Before exposing `/api/v1/me/*` or either chat stream, configure non-empty
`ONTOAGENT_INTERNAL_API_TOKEN`, `ONTOAGENT_USER_CONTEXT_SECRET`, and a valid
`ONTOAGENT_ENCRYPTION_KEY_CURRENT`. Clients must send the internal token as
`X-Internal-Token`; `/api/v1/me/*` requests must also carry signed user-context headers.

Sample session:
```
$ python -m ontoportal_agent.cli chat
OntoPortal Agent ready. Type 'exit' to quit.

user> Summarize the ontology coverage for cancer staging.
agent> Provides an answer with citations from the RAG service...
```

### Publishing prepared artifacts
When the agent prepares an ontology workspace, review the files under
`<ONTOLOGY_WORKDIR>/<workspace>`. Publish manually with:

```bash
python -m ontoportal_agent.cli publish \
  NCIT \
  /tmp/ontoportal-agent/ncit-updates/ncit.ttl \
  --contact-email you@example.org \
  --notes "Fixing tumor stage hierarchy"
```

Set `ONTOAGENT_REQUIRE_MANUAL_APPROVAL=false` to allow the agent to call the publish command
automatically after sandbox execution.

### Working with MCP tools
The agent loads MCP endpoints from `ONTOAGENT_MCP_ENDPOINTS` (comma-separated). Each endpoint must
expose `/tools` and `/invoke`. The default points to the `ontoportal-rag-mcp` MCP adapter at
`http://localhost:8000/mcp`. Configure `ONTOAGENT_MCP_API_KEY` when the MCP endpoint is protected.
Use `ONTOAGENT_MCP_RAG_TOOL_NAME` if your MCP server exposes retrieval under a different tool name.
Once configured, MCP tools appear in planning outputs and can be
invoked from sandbox code or follow-up actions.

---

## Workspaces & Artifacts
- Workspaces live under `ONTOAGENT_ONTOLOGY_WORKDIR` (default `/tmp/ontoportal-agent`).
- Each planning action may create or update `.ttl`, `.rdf`, or `.owl` artifacts.
- `OntologyRepository.save_graph(graph, workspace, filename)` persists changes and returns an
  `OntologyArtifact`. Include this call at the end of your sandbox code to ensure edits are saved.
- Run ROBOT, SHACL, or custom validators inside the sandbox to produce auditable logs that the
  reviewer can inspect before publishing.

---

## Developer Guide

### Project layout
```
ontoportal-agent/
├── pyproject.toml          # Packaging metadata & dependencies
├── src/ontoportal_agent/   # Runtime package
│   ├── agent/              # LangGraph wiring & state definitions
│   ├── cli.py              # Typer-based CLI entry points
│   ├── config.py           # Pydantic settings
│   ├── ontology_repository.py
│   ├── sandbox.py
│   ├── publishing.py
│   └── rag_client.py
└── tests/                  # Pytest suite
```

Add new functionality inside `src/ontoportal_agent` and expose it through the `OntoPortalAgent`
runtime or CLI commands as needed.

### Extending the LangGraph
1. Inspect `agent/graph.py` to understand the current state machine.
2. Add new states or conditional branches via `graph.add_node`, `graph.add_edge`, and
   `graph.add_conditional_edges`.
3. Update `AgentState` (in `agent/state.py`) to include any additional fields you need to carry
   between steps.
4. Prefer pure functions for node implementations to keep the state transitions testable.

### Creating custom tools
- Reuse the `PythonSandbox` or build `langchain_core.tools.StructuredTool` instances in
  `agent/tools.py`. Example: wrap ROBOT, SHACL, or custom data fetchers.
- Register new tools in your LangGraph plan or expose them through MCP endpoints so the LLM can
  call them during planning.

### Testing
Install test extras and run Pytest:
```bash
pip install -e .[test]
pytest
```
`tests/test_tensile_ontology.py` demonstrates how to stub external systems (LLM, ROBOT CLI, REST
API) to exercise the entire plan/execute/publish loop end-to-end.

### Troubleshooting tips
- **Missing citations or slow responses** - Verify the `ontoportal-rag-mcp` FastAPI service is running
  and reachable at `ONTOAGENT_RAG_BASE_URL`.
- **Sandbox failures** - The agent captures stdout from each action; inspect the summary in the
  approval message or open the generated artifacts directly.
- **Publication errors** - Check the OntoPortal API response body (logged by the CLI) and confirm
  the API key has submission privileges for the target ontology acronym.

---

## Versioning & Roadmap
This package is versioned independently (see `pyproject.toml`). Planned enhancements include
adopting LangGraph checkpoints, richer MCP tool discovery, and deeper integration with the
OntoPortal change management UI.

For feature requests or bug reports, open an issue in the main OntoPortal repository and tag it
with `component: agent`.
