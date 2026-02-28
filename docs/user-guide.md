# OntoPortal Agent User Guide

This guide walks through installing the agent, configuring credentials, running interactive sessions,
publishing ontology updates, and extending capabilities with Model Context Protocol endpoints.

## Prerequisites
- Python >= 3.11
- Access to the OntoPortal RAG FastAPI service (see the standalone `ontoportal-rag-mcp` project)
- OntoPortal REST API key with submission privileges
- OpenAI-compatible chat completion model (defaults to `gpt-4o-mini`)

## Installation
```bash
cd /home/todor/ontoportal/ontoportal-agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment Configuration
Create `.env` alongside `pyproject.toml` and set the variables you need. Prefixes must use
`ONTOAGENT_` (the aliases listed below are accepted as environment keys as well).

| Variable | Alias | Default | Purpose |
| -------- | ----- | ------- | ------- |
| `ONTOAGENT_OPENAI_API_KEY` | `OPENAI_API_KEY` | _required_ | API key for the OpenAI-compatible LLM. |
| `ONTOAGENT_OPENAI_API_BASE` | `OPENAI_API_BASE` | `https://api.openai.com/v1` | Base URL for chat completion API. |
| `ONTOAGENT_LLM_MODEL` | `LLM_MODEL` | `gpt-4o-mini` | Chat-completion model used for planning and responses. |
| `ONTOAGENT_RAG_BASE_URL` | `RAG_BASE_URL` | `http://localhost:8000` | Base URL of the OntoPortal RAG FastAPI service. |
| `ONTOAGENT_RAG_QUERY_PATH` | `RAG_QUERY_PATH` | `/api/v1/query` | Relative path for the RAG query endpoint. |
| `ONTOAGENT_ONTOPORTAL_API_KEY` | `ONTOPORTAL_API_KEY` | _required_ | API key for the OntoPortal REST API. |
| `ONTOAGENT_ONTOLOGY_WORKDIR` | `ONTOLOGY_WORKDIR` | `/tmp/ontoportal-agent` | Workspace root for sandbox artifacts. |
| `ONTOAGENT_REQUIRE_MANUAL_APPROVAL` | `REQUIRE_MANUAL_APPROVAL` | `true` | Forces manual review before publishing. |
| `ONTOAGENT_MCP_ENDPOINTS` | `MCP_ENDPOINTS` | `<RAG_BASE_URL>/mcp` | Comma-separated list of MCP base URLs. |
| `ONTOAGENT_MCP_API_KEY` | `MCP_API_KEY` | _(unset)_ | Optional shared API key sent as `X-API-Key` for protected MCP endpoints. |
| `ONTOAGENT_MCP_RAG_TOOL_NAME` | `MCP_RAG_TOOL_NAME` | `rag_query` | MCP tool name used for retrieval before HTTP fallback. |
| `ONTOAGENT_INTERNAL_API_TOKEN` | `INTERNAL_API_TOKEN` | _(unset)_ | Optional shared secret expected as `X-Internal-Token` by the streaming API. |

Example `.env`:

```dotenv
ONTOAGENT_OPENAI_API_KEY=sk-...
ONTOAGENT_ONTOPORTAL_API_KEY=matportal-key
ONTOAGENT_RAG_BASE_URL=http://localhost:8000
ONTOAGENT_REQUIRE_MANUAL_APPROVAL=true
ONTOAGENT_MCP_ENDPOINTS=http://localhost:8000/mcp
ONTOAGENT_MCP_API_KEY=change-me
ONTOAGENT_INTERNAL_API_TOKEN=change-me
```

## Launch the Chat Interface
```bash
python -m ontoportal_agent.cli chat
```

## Launch the Streaming API
```bash
python -m ontoportal_agent.server
```

- Health check: `GET /healthz`
- UI stream endpoint: `POST /api/v1/chat/stream`

Sample session:

```
$ python -m ontoportal_agent.cli chat
OntoPortal Agent ready. Type 'exit' to quit.

user> Summarize the ontology coverage for cancer staging.
agent> Provides an answer with citations from the RAG service...
```

## End-to-End Editing Walkthrough
1. Ask for an edit:  
   `user> Create a new polymer tensile test ontology, validate it with ROBOT, and submit privately.`
2. Review the generated plan in the agent's response. It lists the workspace, actions, and suggested
   publication payload.
3. Inspect artifacts under `<ONTOLOGY_WORKDIR>/<workspace>`; e.g. `/tmp/ontoportal-agent/polymer/tensile.ttl`.
4. Run additional validators locally if desired (ROBOT, SHACL, etc.).
5. Publish when satisfied:
   ```bash
   python -m ontoportal_agent.cli publish \
     POLYTENS /tmp/ontoportal-agent/polymer/tensile.ttl \
     --contact-email you@example.org \
     --notes "Private tensile test ontology"
   ```
6. The command prints the OntoPortal submission ID. Track it through the MatPortal UI.

To enable unattended publishing set `ONTOAGENT_REQUIRE_MANUAL_APPROVAL=false`. The agent will submit
artifacts automatically after sandbox execution and publish-phase validation.

## Working with MCP Tools
- Define additional endpoints via `ONTOAGENT_MCP_ENDPOINTS=http://example.com/mcp,http://other/mcp`.
- Each endpoint must expose `/tools` and `/invoke` in the same style as the `ontoportal-rag-mcp`
  implementation.
- Set `ONTOAGENT_MCP_API_KEY` when MCP endpoints enforce key-based authentication.
- The agent lists available tools during plan generation. Leverage them in follow-up prompts or
  inside sandbox code to tap into downstream services.

## Publishing Prepared Artifacts Manually
When manual approval is required, the agent response includes the workspace name, sandbox output, and
action summaries. After reviewing:

```bash
python -m ontoportal_agent.cli publish \
  NCIT /tmp/ontoportal-agent/ncit-updates/ncit.ttl \
  --contact-email you@example.org \
  --notes "Fixing tumor stage hierarchy"
```

Use `--private` to submit artifacts as private MatPortal submissions.
