# OntoPortal Agent Operations

Operational guidance for running the agent in shared environments, maintaining workspaces, and
integrating with broader OntoPortal infrastructure.

## Workspace & Artifact Management
- Workspaces live under `ONTOAGENT_ONTOLOGY_WORKDIR` (default `/tmp/ontoportal-agent`).
- Each change plan action may create or update `.ttl`, `.rdf`, or `.owl` artifacts within the
  workspace directory. Files outside this tree are never touched.
- `OntologyRepository.save_graph(graph, workspace, filename)` serializes changes and returns an
  `OntologyArtifact`. Ensure sandbox code calls this helper after modifications so reviewers can
  inspect the output.
- Use source control or object storage to back up workspace directories if long-running reviews are
  expected. Copy the entire workspace folder to preserve context and validation logs.

## Security & Safety Considerations
- The sandbox executes Python code with access to rdflib, owlready2, and the standard library.
  While intended for trusted collaborators, treat it as semi-privileged code execution.
- OpenCode runtime command blocking is enabled by default (`ONTOAGENT_OPENCODE_BLOCK_DANGEROUS_COMMANDS=true`).
  This terminates runs that attempt package installation, privilege escalation, remote shell piping, or
  host-level destructive commands.
- Keep `ONTOAGENT_REQUIRE_MANUAL_APPROVAL=true` in environments where user prompts come from
  untrusted sources. This prevents automated publication of malicious or incorrect ontologies.
- Review sandbox stdout carefully for warnings emitted by ROBOT or other validators and verify that
  only intended files are written before publishing.
- Rotate OntoPortal API keys regularly and scope them to the ontologies the agent is allowed to
  submit.

## Running as a Service
- Wrap `OntoPortalAgent` in a long-lived process (e.g. FastAPI, Slack bot, or CLI daemon) to expose
  automated workflows. Cache settings with `ontoportal_agent.config.get_settings()` as the CLI does.
- Configure a persistent workspace directory (e.g. `/var/lib/ontoportal-agent`) instead of `/tmp`
  so interim artifacts survive restarts.
- Use process supervision (systemd, Docker, Kubernetes) and inject configuration via environment
  variables managed by your secrets store.
- Log agent responses, sandbox stdout, and OntoPortal submission IDs for auditability.

## CI/CD Integration
- Add smoke tests that run `pytest -k agent` to validate change plans after dependency updates.
- For automated ontology migrations, script prompts or directly call `OntoPortalAgent.invoke()` with
  prepared instructions. Store generated workspaces as pipeline artifacts so reviewers can inspect
  them before promoting to production.
- When autopublish is enabled in CI, gate deployments behind manual approvals at the pipeline level
  to mimic the interactive review.

## RAG Index Coordination
- The agent depends on the OntoPortal RAG FastAPI service for citations. Coordinate with the RAG
  ingestion team to ensure ontology updates are re-indexed after publication.
- For local testing, run the ingestion pipeline against the modified ontology and restart the RAG
  service before prompting the agent to verify the changes appear in retrieval results.
- Document the ingestion cadence (e.g. nightly rebuilds) so ontology authors know when their updates
  will surface in answers.
