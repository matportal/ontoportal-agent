from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import requests

from .config import get_settings


@dataclass
class RagChunk:
    ontology_id: str
    version: str
    content: str
    metadata: dict[str, Any]


@dataclass
class RagResult:
    answer: str
    sources: List[RagChunk]


class RagClient:
    """Simple HTTP client for the OntoPortal-RAG FastAPI service."""

    def __init__(self, *, base_url: str | None = None, query_path: str | None = None):
        settings = get_settings()
        self.base_url = base_url or settings.rag_base_url
        self.query_path = query_path or settings.rag_query_path

    def query(self, question: str, *, top_k: int | None = None) -> RagResult:
        url = f"{self.base_url.rstrip('/')}{self.query_path}"
        payload: dict[str, Any] = {"query": question}
        if top_k is not None:
            payload["top_k"] = int(top_k)
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        payload = response.json()
        sources = [
            RagChunk(
                ontology_id=item.get("ontology_id", "unknown"),
                version=item.get("version", "unknown"),
                content=item.get("content", ""),
                metadata=item.get("metadata", {}),
            )
            for item in payload.get("sources", [])
        ]
        return RagResult(answer=payload.get("answer", ""), sources=sources)

    def graph_query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        ontology_id: str | None = None,
    ) -> RagResult:
        url = f"{self.base_url.rstrip('/')}/api/v1/graph-query"
        payload: dict[str, Any] = {"query": question}
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if ontology_id is not None:
            payload["ontology_id"] = ontology_id
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        payload = response.json()
        sources = []
        for item in payload.get("sources", []):
            metadata = dict(item.get("metadata", {}) or {})
            for key in ("id", "kind", "source_locator", "citation_text", "entity_iri", "named_graph", "authority_level", "score"):
                if key in item and item[key] is not None:
                    metadata.setdefault(key, item[key])
            sources.append(
                RagChunk(
                    ontology_id=item.get("ontology_id") or "unknown",
                    version=item.get("version") or "unknown",
                    content=item.get("content", ""),
                    metadata=metadata,
                )
            )
        return RagResult(answer=payload.get("answer", ""), sources=sources)
