from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from rdflib import Graph

from ..config import get_settings
from ..mcp_client import McpClient, McpInvocationError
from ..ontology_repository import OntologyRepository, OntologyArtifact
from ..publishing import OntoPortalPublisher
from ..rag_client import RagClient
from ..sandbox import PythonSandbox
from .state import AgentState


def _extract_generation_usage(reply: Any) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}

    usage_metadata = getattr(reply, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage["prompt_tokens"] = usage_metadata.get("input_tokens")
        usage["completion_tokens"] = usage_metadata.get("output_tokens")
        usage["total_tokens"] = usage_metadata.get("total_tokens")
        output_details = usage_metadata.get("output_token_details")
        if isinstance(output_details, dict):
            usage["reasoning_tokens"] = output_details.get("reasoning")

    response_metadata = getattr(reply, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            usage["prompt_tokens"] = usage.get("prompt_tokens") or token_usage.get("prompt_tokens")
            usage["completion_tokens"] = usage.get("completion_tokens") or token_usage.get("completion_tokens")
            usage["total_tokens"] = usage.get("total_tokens") or token_usage.get("total_tokens")

            completion_details = token_usage.get("completion_tokens_details")
            if isinstance(completion_details, dict):
                usage["reasoning_tokens"] = usage.get("reasoning_tokens") or completion_details.get("reasoning_tokens")
            usage["reasoning_tokens"] = usage.get("reasoning_tokens") or token_usage.get("reasoning_tokens")
        usage["model"] = response_metadata.get("model_name") or response_metadata.get("model")

    return {key: value for key, value in usage.items() if value is not None}


def build_agent_graph(repository: OntologyRepository) -> StateGraph[AgentState]:
    settings = get_settings()
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        model=settings.llm_model,
        temperature=0.0,
    )
    mcp_client = McpClient(
        settings.resolved_mcp_endpoints(),
        api_key=settings.mcp_api_key,
    )
    rag_client = RagClient()
    publisher = OntoPortalPublisher()
    sandbox = PythonSandbox(repository)

    graph = StateGraph(AgentState)

    def classify_intent(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Classify the user's intent as either RETRIEVE or EDIT. Respond with a single word.",
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm
        intent = chain.invoke({"question": state["user_input"]}).content.strip().upper()
        state["intent"] = "EDIT" if "EDIT" in intent else "RETRIEVE"
        return state

    def retrieve_answer(state: AgentState) -> AgentState:
        question = state["user_input"]

        try:
            mcp_payload = mcp_client.invoke_rag_query(
                question,
                tool_name=settings.mcp_rag_tool_name,
            )
            sources = mcp_payload.get("sources", [])
            state["rag_result"] = mcp_payload.get("answer", "")
            state["citations"] = [
                f"{src.get('ontology_id', 'unknown')} v{src.get('version', 'unknown')}"
                for src in sources
            ]
            state["retrieval_backend"] = "mcp"
            return state
        except (McpInvocationError, KeyError, TypeError, ValueError) as err:
            state["retrieval_backend"] = "rag-http-fallback"
            state["retrieval_error"] = str(err)

        try:
            result = rag_client.query(question)
            state["rag_result"] = result.answer
            state["citations"] = [f"{src.ontology_id} v{src.version}" for src in result.sources]
        except Exception as err:  # noqa: BLE001 - we intentionally degrade to non-RAG response.
            state["retrieval_backend"] = "none"
            existing_error = state.get("retrieval_error")
            if existing_error:
                state["retrieval_error"] = f"{existing_error}; fallback failed: {err}"
            else:
                state["retrieval_error"] = str(err)
            state["rag_result"] = ""
            state["citations"] = []
        return state

    def generate_response(state: AgentState) -> AgentState:
        question = state["user_input"]
        rag_result = state.get("rag_result", "") or ""
        if len(rag_result) > settings.max_rag_context_chars:
            rag_result = rag_result[: settings.max_rag_context_chars]

        citations = state.get("citations", [])
        citation_text = "\n".join(f"- {c}" for c in citations) if citations else "- none"
        retrieval_backend = state.get("retrieval_backend", "unknown")
        retrieval_error = state.get("retrieval_error", "")
        messages = [
            SystemMessage(
                content=(
                    "You are the MatPortal ontology assistant.\n"
                    "Rules:\n"
                    "- Be concise and readable.\n"
                    "- Use short paragraphs or bullets only.\n"
                    "- Do not use markdown tables unless the user explicitly asks for a table.\n"
                    "- Do not include code blocks unless the user asks for code.\n"
                    "- Never invent ontology names, class URIs, or facts not present in the retrieved context.\n"
                    "- If evidence is weak or missing, state uncertainty clearly.\n"
                    "- For recommendation questions, return up to 3 options with a one-line reason each."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n"
                    f"Retrieved context: {rag_result}\n"
                    f"Retrieval backend: {retrieval_backend}\n"
                    f"Retrieval error: {retrieval_error}\n"
                    f"Citations:\n{citation_text}\n"
                    "Write a direct answer for the user."
                )
            ),
        ]
        reply = llm.invoke(messages)
        llm_content = reply.content if isinstance(reply.content, str) else str(reply.content or "")
        state["generation_usage"] = _extract_generation_usage(reply)
        if llm_content.strip():
            if len(llm_content) > settings.max_response_chars:
                llm_content = llm_content[: settings.max_response_chars].rstrip() + "..."
            state["final_response"] = llm_content
            state["generation_backend"] = f"llm:{settings.llm_model}"
        else:
            state["final_response"] = ""
            state["generation_backend"] = "none"
            state["generation_error"] = "LLM returned empty content"
        return state

    def plan_edit(state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a senior ontology engineer. Produce a JSON plan describing how to modify or create ontologies "
                        "using rdflib/owlready2 inside a Python sandbox. Structure the response as: "
                        "{{\"workspace\": str, \"actions\": ["
                        "{{\"description\": str, \"artifact\": str, \"create\": bool, \"format\": \"turtle\"|\"rdf\", \"code\": str}}"
                        "], \"publish\": {{\"acronym\": str, \"artifact\": str, \"contact_email\": str, \"notes\": str, \"private\": bool}}}}."
                        "The code must use the provided `graph`, `artifact`, and `ontology_repo` objects, and after modifications call "
                        "`ontology_repo.save_graph(graph, workspace, artifact)` to persist changes."
                    ),
                ),
                ("human", "{question}"),
            ]
        )
        response = (prompt | llm).invoke({"question": state["user_input"]}).content
        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            plan = {
                "workspace": "session",
                "actions": [
                    {
                        "description": "No action generated; placeholder plan.",
                        "artifact": "output.ttl",
                        "create": True,
                        "format": "turtle",
                        "code": (
                            "from rdflib import Graph\n"
                            "# Modify graph here\n"
                            "ontology_repo.save_graph(graph, workspace, 'output.ttl')"
                        ),
                    }
                ],
                "publish": None,
            }

        workspace_name = plan.get("workspace", "session-workspace")
        state["workspace"] = workspace_name
        state["plan_actions"] = plan.get("actions", [])
        state["publish_payload"] = plan.get("publish")
        state["change_notes"] = [action.get("description", "") for action in state.get("plan_actions", [])]
        state["approval_required"] = True
        return state

    def execute_actions(state: AgentState) -> AgentState:
        actions: List[Dict[str, Any]] = state.get("plan_actions", [])
        workspace_name = state.get("workspace", "session-workspace")
        workspace = repository.create_workspace(workspace_name)

        outputs: List[str] = []
        for action in actions:
            artifact_name = action.get("artifact", "output.ttl")
            create = bool(action.get("create", False))
            fmt = action.get("format", "turtle")
            artifact_path = workspace / artifact_name

            if create or not artifact_path.exists():
                Graph().serialize(destination=str(artifact_path), format=fmt)

            artifact = OntologyArtifact(path=artifact_path, format="ttl" if fmt == "turtle" else fmt)
            graph = repository.load_graph(artifact)

            code = action.get("code", "")
            result = sandbox.run(
                code=code,
                graph=graph,
                artifact=artifact,
                extra_globals={"workspace": workspace},
            )
            outputs.append(result.stdout.strip())

        state["sandbox_output"] = "\n---\n".join(filter(None, outputs)) or "Sandbox executed without output."
        state["workspace"] = workspace_name
        return state

    def approval_gate(state: AgentState) -> AgentState:
        workspace = state.get("workspace", "workspace")
        artifact_hint = None
        publish_payload = state.get("publish_payload")
        if publish_payload:
            artifact_hint = publish_payload.get("artifact")

        summary_lines = state.get("change_notes", [])
        sandbox_output = state.get("sandbox_output", "")
        summary = "\n".join(summary_lines) if summary_lines else "- No change notes generated."
        instructions = (
            f"Workspace: {workspace}.\n"
            f"Sandbox output:\n{sandbox_output}\n"
            "Review the changes. When satisfied, publish with: \n"
            "`ontoportal-agent publish --acronym <ACRONYM> --artifact <PATH> --contact-email <EMAIL>`"
        )
        if artifact_hint:
            artifact_path = Path(repository.workdir / workspace / artifact_hint)
            instructions += f"\nSuggested artifact: {artifact_path}"

        auto_publish = bool(not settings.require_manual_approval and publish_payload)
        state["auto_publish"] = auto_publish

        if auto_publish:
            state["final_response"] = "Publishing prepared ontology automatically."
        else:
            state["final_response"] = (
                "Proposed ontology edits (pending approval):\n"
                f"{summary}\n\n{instructions}"
            )
        return state

    def publish_changes(state: AgentState) -> AgentState:
        meta = state.get("publish_payload")
        workspace = state.get("workspace", "workspace")
        if meta:
            artifact_rel = meta["artifact"]
            artifact_path = Path(repository.workdir / workspace / artifact_rel)
            publisher.submit_ontology(
                acronym=meta["acronym"],
                artifact_path=artifact_path,
                contact_email=meta["contact_email"],
                notes=meta.get("notes", "Submitted via agent"),
                is_private=bool(meta.get("private")),
            )
            state["final_response"] = "Ontology update submitted to OntoPortal."
        return state

    graph.add_node("classify", classify_intent)
    graph.add_node("retrieve", retrieve_answer)
    graph.add_node("respond", generate_response)
    graph.add_node("plan_edit", plan_edit)
    graph.add_node("execute_actions", execute_actions)
    graph.add_node("await_approval", approval_gate)
    graph.add_node("publish", publish_changes)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: state["intent"],
        {
            "RETRIEVE": "retrieve",
            "EDIT": "plan_edit",
        },
    )
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)
    graph.add_edge("plan_edit", "execute_actions")
    graph.add_edge("execute_actions", "await_approval")
    graph.add_conditional_edges(
        "await_approval",
        lambda state: "publish" if state.get("auto_publish") else "end",
        {
            "publish": "publish",
            "end": END,
        },
    )
    graph.add_edge("publish", END)

    return graph
