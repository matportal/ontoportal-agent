from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRuntimeOptions:
    openai_api_key: str
    openai_api_base: str | None = None
    llm_model: str | None = None
    rag_top_k: int | None = None
    rag_base_url: str | None = None
    rag_query_path: str | None = None
    mcp_endpoints: list[str | dict[str, Any]] = field(default_factory=list)
    mcp_api_key: str | None = None
    mcp_rag_tool_name: str | None = None
