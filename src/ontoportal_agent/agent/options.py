from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRuntimeOptions:
    openai_api_key: str
    generation_provider: str | None = None
    generation_api_key_configured: bool = False
    openai_api_base: str | None = None
    llm_model: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None
    vertex_service_account_json: str | None = None
    rag_top_k: int | None = None
    rag_base_url: str | None = None
    rag_query_path: str | None = None
    mcp_endpoints: list[str | dict[str, Any]] = field(default_factory=list)
    mcp_api_key: str | None = None
    mcp_rag_tool_name: str | None = None
    opencode_auth_source: str = "auto"
    opencode_auth_kind: str | None = None
    opencode_auth_json: str | None = None
    codex_auth_json: str | None = None
