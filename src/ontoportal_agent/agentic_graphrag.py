from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from .rag_client import RagChunk, RagClient, RagResult

_NEGATIVE_ANSWER_MARKERS = (
    "no evidence",
    "not enough information",
    "insufficient evidence",
    "not found",
    "no sources",
    "unable to answer",
)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model text, including fenced responses."""
    clean = str(text or "").strip()
    if not clean:
        return {}

    try:
        payload = json.loads(clean)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", clean, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            payload = json.loads(fence_match.group(1).strip())
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

    start = clean.find("{")
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(clean)):
        char = clean[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(clean[start : index + 1].strip())
                    return payload if isinstance(payload, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _stringify_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value or "")


def _invoke_json(llm: Any | None, prompt: str) -> dict[str, Any]:
    if llm is None:
        return {}
    try:
        reply = llm.invoke([HumanMessage(content=prompt)])
    except Exception:
        return {}
    return _extract_json(_stringify_message_content(getattr(reply, "content", reply)))


def _coerce_information_needs(question: str, needs: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if needs:
        cleaned = [" ".join(str(item or "").split()) for item in needs]
        cleaned = [item for item in cleaned if item]
        if cleaned:
            return list(dict.fromkeys(cleaned))[:8]

    question_text = " ".join(str(question or "").split())
    if not question_text:
        return [""]

    fragments = re.split(r"\s*(?:\?|;|\b(?:and|then|also)\b)\s*", question_text, flags=re.IGNORECASE)
    cleaned = [fragment.strip(" .,:;?\n\t") for fragment in fragments if len(fragment.strip()) >= 12]
    if not cleaned:
        return [question_text]
    return list(dict.fromkeys(cleaned))[:8]


def _decompose_with_model(question: str, llm: Any | None, fallback_needs: list[str]) -> list[str]:
    prompt = (
        "Decompose this MatPortal ontology question into atomic information needs for a bounded GraphRAG retrieval loop.\n"
        "Each need must be self-contained and answerable from ontology evidence such as labels, IRIs, class hierarchy, "
        "domain/range constraints, mappings, documentation chunks, or entity cards.\n\n"
        f"Question: {question}\n\n"
        "Return only JSON: {\"needs\": [\"need 1\", \"need 2\"]}"
    )
    parsed = _invoke_json(llm, prompt)
    raw_needs = parsed.get("needs") if isinstance(parsed, dict) else None
    return _coerce_information_needs(question, raw_needs if isinstance(raw_needs, list) else fallback_needs)


def _source_citation(source: RagChunk, index: int) -> str:
    metadata = source.metadata or {}
    for key in ("citation_text", "id", "source_locator", "entity_iri", "named_graph", "chunk_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    label = f"{source.ontology_id} v{source.version}".strip()
    return label if label != "unknown vunknown" else f"source-{index + 1}"


def _source_key(source: RagChunk) -> str:
    metadata = source.metadata or {}
    for key in ("id", "source_locator", "entity_iri", "named_graph", "chunk_id"):
        value = metadata.get(key)
        if value:
            return f"{source.ontology_id}:{source.version}:{key}:{value}"
    return f"{source.ontology_id}:{source.version}:{source.content[:512]}"


def _source_payload(source: RagChunk, index: int) -> dict[str, Any]:
    return {
        "ontology_id": source.ontology_id,
        "version": source.version,
        "content": source.content,
        "metadata": source.metadata,
        "citation": _source_citation(source, index),
    }


def _has_negative_answer_marker(answer: str) -> bool:
    lower = str(answer or "").lower()
    return any(marker in lower for marker in _NEGATIVE_ANSWER_MARKERS)


def _fallback_rewrite(*, original_question: str, need: str, iteration: int) -> str:
    if iteration <= 1:
        return (
            f"Find canonical ontology evidence for: {need}. Include exact labels, IRIs, definitions, hierarchy, "
            "domain/range constraints, mappings, synonyms, and named-graph provenance."
        )
    return (
        f"Original question: {original_question}\n"
        f"Missing information need: {need}\n"
        "Search with alternative labels, synonyms, ontology IDs, parent/child terms, property constraints, and mapping terms."
    )


def _assess_attempt(
    *,
    need: str,
    result: RagResult,
    llm: Any | None,
    current_query: str,
    original_question: str,
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    has_sources = bool(result.sources)
    has_answer = bool(str(result.answer or "").strip()) and not _has_negative_answer_marker(result.answer)
    fallback_status = "satisfied" if has_sources and has_answer else "unsatisfied"
    fallback_assessment = (
        "Answer and evidence sources were returned." if fallback_status == "satisfied" else "Evidence sources or usable answer are missing."
    )

    sources_summary = "\n".join(
        f"- citation={_source_citation(source, index)} ontology={source.ontology_id} version={source.version} "
        f"metadata={source.metadata} content={source.content[:1000]}"
        for index, source in enumerate(result.sources)
    ) or "(no sources returned)"

    prompt = (
        "Evaluate whether retrieved MatPortal GraphRAG evidence is sufficient for one atomic information need.\n"
        "A need is satisfied only if the answer is supported by returned evidence sources. If sources are absent, mark it unsatisfied.\n\n"
        f"Original user question: {original_question}\n"
        f"Atomic information need: {need}\n"
        f"Current query: {current_query}\n"
        f"Attempt: {iteration} of {max_iterations}\n\n"
        f"Retrieved answer:\n{result.answer or '(empty)'}\n\n"
        f"Retrieved sources:\n{sources_summary}\n\n"
        "Return only JSON: {\"status\": \"satisfied\"|\"partially_satisfied\"|\"unsatisfied\", "
        "\"assessment\": \"short reason\", \"missing_info\": \"remaining gap or none\", "
        "\"suggested_rewrite\": \"next query or null\"}"
    )
    parsed = _invoke_json(llm, prompt)

    status = str(parsed.get("status") or fallback_status).strip().lower()
    if status not in {"satisfied", "partially_satisfied", "unsatisfied"}:
        status = fallback_status
    if status == "satisfied" and not (has_sources and has_answer):
        status = "unsatisfied"

    suggested_rewrite = parsed.get("suggested_rewrite")
    if status != "satisfied" and iteration < max_iterations:
        suggested_rewrite = str(suggested_rewrite or "").strip() or _fallback_rewrite(
            original_question=original_question,
            need=need,
            iteration=iteration,
        )
        if suggested_rewrite == current_query:
            suggested_rewrite = _fallback_rewrite(original_question=original_question, need=need, iteration=iteration + 1)
    else:
        suggested_rewrite = None

    return {
        "status": status,
        "assessment": str(parsed.get("assessment") or fallback_assessment),
        "missing_info": str(parsed.get("missing_info") or ("None" if status == "satisfied" else "More ontology evidence is needed.")),
        "suggested_rewrite": suggested_rewrite,
    }


def _build_final_context(question: str, coverage: list[dict[str, Any]], attempts: list[dict[str, Any]], sources: list[dict[str, Any]]) -> str:
    source_index = {source["citation"]: source for source in sources}
    lines = [f"Agentic GraphRAG evidence for: {question}"]
    for item in coverage:
        lines.append("")
        lines.append(f"Need: {item['need']}")
        lines.append(f"Status: {item['status']}")
        answer = str(item.get("answer") or "").strip()
        if answer:
            citations = item.get("citations") or []
            citation_text = ", ".join(str(citation) for citation in citations if citation)
            lines.append(f"Evidence-backed answer: {answer}{f' [{citation_text}]' if citation_text else ''}")
        if item.get("status") != "satisfied":
            lines.append(f"Gap: {item.get('missing_info') or 'More evidence is needed.'}")

    if source_index:
        lines.append("")
        lines.append("Sources:")
        for citation, source in source_index.items():
            locator = source.get("metadata", {}).get("source_locator") or source.get("metadata", {}).get("entity_iri") or ""
            label = f"{source.get('ontology_id')} v{source.get('version')}"
            lines.append(f"- {citation}: {label}{f' — {locator}' if locator else ''}")

    if not attempts:
        lines.append("")
        lines.append("No retrieval attempts were completed.")
    return "\n".join(lines).strip()


def run_agentic_graphrag(
    question: str,
    rag_client: RagClient,
    llm: Any | None = None,
    *,
    ontology_id: str | None = None,
    top_k: int | None = None,
    strict_scope: bool = True,
    allow_scope_expansion: bool = False,
    max_iterations: int = 3,
    information_needs: list[str] | None = None,
) -> dict[str, Any]:
    """Run a bounded sufficient-evidence loop over the deterministic GraphRAG API.

    The loop may use an LLM for decomposition and gap analysis, but all evidence
    acquisition remains constrained to RagClient.graph_query. It never generates
    or executes raw SPARQL and never expands ontology scope unless the caller
    explicitly passes allow_scope_expansion=True.
    """
    clean_question = " ".join(str(question or "").split())
    iteration_limit = max(1, min(int(max_iterations or 1), 5))
    normalized_top_k = None if top_k is None else max(1, min(int(top_k), 50))
    fallback_needs = _coerce_information_needs(clean_question, information_needs)
    needs = _decompose_with_model(clean_question, llm, fallback_needs) if llm is not None else fallback_needs

    coverage: list[dict[str, Any]] = []
    gaps: list[str] = []
    attempts: list[dict[str, Any]] = []
    unique_sources: dict[str, RagChunk] = {}

    for need in needs:
        current_query = need
        last_assessment: dict[str, Any] = {
            "status": "unsatisfied",
            "assessment": "No retrieval attempt completed.",
            "missing_info": "Not queried yet.",
            "suggested_rewrite": None,
        }
        best_answer = ""
        best_citations: list[str] = []
        attempts_count = 0

        for iteration in range(1, iteration_limit + 1):
            attempts_count += 1
            attempt: dict[str, Any] = {
                "need": need,
                "iteration": iteration,
                "query": current_query,
                "top_k": normalized_top_k,
                "ontology_id": ontology_id,
                "strict_scope": strict_scope,
                "allow_scope_expansion": allow_scope_expansion,
            }
            try:
                result = rag_client.graph_query(
                    current_query,
                    top_k=normalized_top_k,
                    ontology_id=ontology_id,
                    strict_scope=strict_scope,
                    allow_scope_expansion=allow_scope_expansion,
                )
            except Exception as exc:  # noqa: BLE001 - failed retrieval is evidence of unavailability, not a crash.
                attempt.update({"answer": "", "sources": [], "error": str(exc)})
                attempts.append(attempt)
                last_assessment = {
                    "status": "unsatisfied",
                    "assessment": f"GraphRAG query failed: {exc}",
                    "missing_info": "Retrieval failed before evidence could be collected.",
                    "suggested_rewrite": None,
                }
                break

            attempt_sources = [_source_payload(source, index) for index, source in enumerate(result.sources)]
            attempt.update({"answer": result.answer, "sources": attempt_sources})
            attempts.append(attempt)

            for source in result.sources:
                unique_sources.setdefault(_source_key(source), source)

            last_assessment = _assess_attempt(
                need=need,
                result=result,
                llm=llm,
                current_query=current_query,
                original_question=clean_question,
                iteration=iteration,
                max_iterations=iteration_limit,
            )

            if result.sources and result.answer and not _has_negative_answer_marker(result.answer):
                best_answer = result.answer
                best_citations = [_source_citation(source, index) for index, source in enumerate(result.sources)]

            if last_assessment["status"] == "satisfied":
                break
            rewrite = last_assessment.get("suggested_rewrite")
            if not rewrite:
                break
            current_query = str(rewrite)

        coverage_item = {
            "need": need,
            "status": last_assessment["status"],
            "attempts_count": attempts_count,
            "last_assessment": last_assessment["assessment"],
            "missing_info": last_assessment["missing_info"],
            "answer": best_answer,
            "citations": best_citations,
        }
        coverage.append(coverage_item)
        if coverage_item["status"] != "satisfied":
            gaps.append(need)

    sources = [_source_payload(source, index) for index, source in enumerate(unique_sources.values())]
    sufficient_context = bool(coverage) and not gaps
    final_context = _build_final_context(clean_question, coverage, attempts, sources)

    return {
        "sufficient_context": sufficient_context,
        "coverage": coverage,
        "gaps": gaps,
        "attempts": attempts,
        "sources": sources,
        "final_context": final_context,
        "metadata": {
            "ontology_id": ontology_id,
            "strict_scope": strict_scope,
            "allow_scope_expansion": allow_scope_expansion,
            "max_iterations": iteration_limit,
            "top_k": normalized_top_k,
            "information_need_count": len(needs),
        },
    }
