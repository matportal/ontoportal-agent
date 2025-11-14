# OntoPortal Agent Overview

`ontoportal-agent` automates ontology discovery, editing, and publication workflows on top of the
OntoPortal / MatPortal ecosystem. It stitches together LangChain, LangGraph, Retrieval Augmented
Generation (RAG), and a controlled Python sandbox so that ontology engineers can collaborate with
large language models while keeping every change auditable.

## Key Capabilities
- **Context-aware answers** - Queries the local OntoPortal RAG service and returns responses with
  explicit citations.
- **Guided editing plans** - Uses LangGraph to classify intents, draft JSON change plans, and capture
  reviewer notes.
- **Python sandbox** - Executes rdflib/owlready2 snippets inside isolated workspaces to stage ontology
  updates safely.
- **Publication pipeline** - Submits vetted ontology artifacts through the OntoPortal REST API once a
  reviewer approves or automation is enabled.
- **Model Context Protocol (MCP)** - Discovers external tools exposed over MCP and adds them to the
  agent's toolbox without code changes.

## Documentation Map
- [Architecture](architecture.md) - State machine design, request lifecycle, and component overview.
- [User Guide](user-guide.md) - Installation, configuration, chat workflow, publishing, and MCP tips.
- [Operations](operations.md) - Workspace management, security considerations, CI/service patterns,
  and RAG index coordination.
- [Developer Guide](developer-guide.md) - Project layout, extending LangGraph, writing custom tools,
  and testing.

If you are onboarding, start with the [User Guide](user-guide.md) and return to the overview when you
need deeper dives into the architecture or operational patterns.
