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

    def query(self, question: str) -> RagResult:
        url = f"{self.base_url.rstrip('/')}{self.query_path}"
        response = requests.post(url, json={"query": question}, timeout=60)
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
