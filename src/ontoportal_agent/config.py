from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Runtime configuration for the OntoPortal agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ONTOAGENT_",
        extra="ignore",
        enable_decoding=False,
    )

    # LLM settings
    openai_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("ONTOAGENT_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_api_base: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_OPENAI_API_BASE", "OPENAI_API_BASE"),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("ONTOAGENT_LLM_MODEL", "LLM_MODEL"),
    )
    max_rag_context_chars: int = Field(
        default=12000,
        validation_alias=AliasChoices("ONTOAGENT_MAX_RAG_CONTEXT_CHARS", "MAX_RAG_CONTEXT_CHARS"),
    )
    max_response_chars: int = Field(
        default=2400,
        validation_alias=AliasChoices("ONTOAGENT_MAX_RESPONSE_CHARS", "MAX_RESPONSE_CHARS"),
    )

    # RAG endpoint
    rag_base_url: str = Field(
        default="http://localhost:8000",
        validation_alias=AliasChoices("ONTOAGENT_RAG_BASE_URL", "RAG_BASE_URL"),
    )
    rag_query_path: str = Field(
        default="/api/v1/query",
        validation_alias=AliasChoices("ONTOAGENT_RAG_QUERY_PATH", "RAG_QUERY_PATH"),
    )

    # OntoPortal REST API
    ontoportal_api_base: str = Field(
        default="https://rest.matportal.org",
        validation_alias=AliasChoices("ONTOAGENT_ONTOPORTAL_API_BASE", "ONTOPORTAL_API_BASE"),
    )
    ontoportal_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("ONTOAGENT_ONTOPORTAL_API_KEY", "ONTOPORTAL_API_KEY"),
    )

    # Workspace
    ontology_workdir: Path = Field(
        default=Path("/tmp/ontoportal-agent"),
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_WORKDIR", "ONTOLOGY_WORKDIR"),
    )

    # Approval gate
    require_manual_approval: bool = Field(
        default=True,
        validation_alias=AliasChoices("ONTOAGENT_REQUIRE_MANUAL_APPROVAL", "REQUIRE_MANUAL_APPROVAL"),
    )

    # Model Context Protocol endpoints (comma-separated)
    mcp_endpoints: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("ONTOAGENT_MCP_ENDPOINTS", "MCP_ENDPOINTS"),
    )
    mcp_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_MCP_API_KEY", "MCP_API_KEY"),
    )
    mcp_rag_tool_name: str = Field(
        default="rag_query",
        validation_alias=AliasChoices("ONTOAGENT_MCP_RAG_TOOL_NAME", "MCP_RAG_TOOL_NAME"),
    )

    # Optional shared secret between UI backend and agent API.
    internal_api_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_INTERNAL_API_TOKEN", "INTERNAL_API_TOKEN"),
    )

    @field_validator("mcp_endpoints", mode="before")
    @classmethod
    def _split_endpoints(cls, value: Optional[str] | List[str]):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def resolved_mcp_endpoints(self) -> List[str]:
        if self.mcp_endpoints:
            return self.mcp_endpoints
        return [self.rag_base_url.rstrip("/") + "/mcp"]


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    return AgentSettings()
