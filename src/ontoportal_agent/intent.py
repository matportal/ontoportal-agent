from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

INTENT_RETRIEVE = "RETRIEVE"
INTENT_EDIT = "EDIT"

_RETRIEVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwhat\b",
        r"\bwhich\b",
        r"\bwho\b",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bcompare\b",
        r"\bsummar(?:ize|y)\b",
        r"\bexplain\b",
        r"\blist\b",
        r"\bshow\b",
        r"\bfind\b",
        r"\bsearch\b",
        r"\banaly[sz]e\b",
        r"\bdescribe\b",
        r"\bmarkdown\b",
        r"\btable\b",
        r"\bjson\b",
        r"\bcode block\b",
        r"\bbullet points?\b",
        r"\bdifferences?\b",
    )
]

_EDIT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(create|build|generate|produce|draft|write)\b.{0,60}\b(new\s+)?(ontology|ontologies|ttl|turtle|rdf|owl|ontology file|artifact)\b",
        r"\b(add|remove|delete|rename|update|modify|change|extend|refactor|patch|revise)\b.{0,60}\b(class|classes|property|properties|ontology|ontologies|triple|triples|axiom|axioms|individual|individuals|label|definition|metadata|submission)\b",
        r"\b(publish|submit|upload)\b.{0,60}\b(ontology|artifact|submission|ttl|turtle|rdf|owl|portal)\b",
        r"\b(private|public)\b.{0,40}\b(submission|publish|ontology)\b",
    )
]

_AMBIGUOUS_EDIT_HINTS = re.compile(
    r"\b(ontology|ontologies|submission|artifact|ttl|turtle|rdf|owl|class|classes|property|properties|axiom|individual)\b",
    re.IGNORECASE,
)


def _flatten_chunk_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_flatten_chunk_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            if key in value:
                return _flatten_chunk_text(value[key])
        return "".join(_flatten_chunk_text(item) for item in value.values())
    text = getattr(value, "text", None)
    if text is not None:
        return _flatten_chunk_text(text)
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _flatten_chunk_text(content)
    return str(value)


def _matches_any(patterns: list[re.Pattern[str]], prompt: str) -> bool:
    return any(pattern.search(prompt) for pattern in patterns)


def _llm_classify_intent(llm: Any, prompt: str) -> str:
    reply = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Route the user request as either RETRIEVE or EDIT.\n"
                    "Return EDIT only when the user explicitly asks to create, modify, rename, delete, publish, submit, "
                    "or generate ontology content, RDF/Turtle/OWL files, or ontology submissions.\n"
                    "Return RETRIEVE for requests to explain, compare, summarize, search, analyze, or format an answer "
                    "in markdown, bullets, JSON, or tables, even when ontologies are mentioned.\n"
                    "Respond with exactly RETRIEVE or EDIT."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )
    content = _flatten_chunk_text(getattr(reply, "content", reply)).strip().upper()
    return INTENT_EDIT if re.search(r"\bEDIT\b", content) else INTENT_RETRIEVE


def classify_user_intent(prompt: str, *, llm: Any | None = None) -> str:
    clean = " ".join(str(prompt or "").split())
    if not clean:
        return INTENT_RETRIEVE

    if _matches_any(_EDIT_PATTERNS, clean):
        return INTENT_EDIT

    if _matches_any(_RETRIEVE_PATTERNS, clean):
        return INTENT_RETRIEVE

    if llm is None or not _AMBIGUOUS_EDIT_HINTS.search(clean):
        return INTENT_RETRIEVE

    try:
        return _llm_classify_intent(llm, clean)
    except Exception:
        return INTENT_RETRIEVE
