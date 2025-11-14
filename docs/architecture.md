# OntoPortal Agent Architecture

The agent is built around a LangGraph state machine that coordinates LLM reasoning, retrieval,
ontology editing, human approvals, and publication. The graph manipulates an `AgentState` typed dict
to pass data between nodes.

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

## Request Lifecycle
1. **Intent classification** — an OpenAI-compatible model labels the prompt as `RETRIEVE` or `EDIT`.
2. **Retrieval path** — the agent queries the OntoPortal-RAG service, collects citations, and crafts
   a natural-language answer for the user.
3. **Edit path** — the LLM returns a JSON change plan containing sandbox actions, artifact names,
   and optional publication metadata.
4. **Sandbox execution** — each action runs inside a temporary workspace managed by
   `OntologyRepository`, guaranteeing artifacts land under `<ONTOLOGY_WORKDIR>/<workspace>/`.
5. **Approval gate** — the agent summarizes changes and pauses for reviewer confirmation. When
   `REQUIRE_MANUAL_APPROVAL=false`, publishing can continue automatically.
6. **Publishing** — `OntoPortalPublisher` encodes the artifact and calls the OntoPortal REST API to
   create a new submission.

## Core Modules

| Module | Responsibility |
| ------ | -------------- |
| `agent/graph.py` | Builds the LangGraph state machine; orchestrates routing, planning, execution, and publication. |
| `agent/state.py` | Defines the TypedDict carried between graph nodes. |
| `agent/runtime.py` | Provides the `OntoPortalAgent` interface for CLI and programmatic callers. |
| `config.py` | Loads environment-driven settings via Pydantic and caches them. |
| `rag_client.py` | Talks to the OntoPortal RAG FastAPI service for answer retrieval. |
| `ontology_repository.py` | Manages workspaces, rdflib serialization, and artifact metadata. |
| `sandbox.py` | Runs user-authored Python snippets inside a constrained environment. |
| `publishing.py` | Integrates with OntoPortal/MatPortal REST endpoints for submissions. |
| `mcp_client.py` | Discovers and invokes tools exposed through Model Context Protocol endpoints. |

Refer to the source in `src/ontoportal_agent/agent/graph.py` for detailed implementation notes and
extension hooks.
