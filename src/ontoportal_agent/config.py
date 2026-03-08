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
        default="gemini-3-flash-preview",
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

    # Assistant persistence + user-context security
    database_url: str = Field(
        default="sqlite:///./ontoportal-agent.db",
        validation_alias=AliasChoices("ONTOAGENT_DATABASE_URL", "DATABASE_URL"),
    )
    encryption_key_current: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_ENCRYPTION_KEY_CURRENT", "ENCRYPTION_KEY_CURRENT"),
    )
    encryption_key_previous: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_ENCRYPTION_KEY_PREVIOUS", "ENCRYPTION_KEY_PREVIOUS"),
    )
    user_context_secret: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_USER_CONTEXT_SECRET", "USER_CONTEXT_SECRET"),
    )
    user_context_ttl_seconds: int = Field(
        default=300,
        validation_alias=AliasChoices("ONTOAGENT_USER_CONTEXT_TTL_SECONDS", "USER_CONTEXT_TTL_SECONDS"),
    )
    history_retention_days: int = Field(
        default=90,
        validation_alias=AliasChoices("ONTOAGENT_HISTORY_RETENTION_DAYS", "HISTORY_RETENTION_DAYS"),
    )

    # Deployment-level defaults for provider settings.
    default_generation_provider: str = Field(
        default="openai_compatible",
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_GENERATION_PROVIDER", "DEFAULT_GENERATION_PROVIDER"),
    )
    default_generation_model: str = Field(
        default="gemini-3-flash-preview",
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_GENERATION_MODEL", "DEFAULT_GENERATION_MODEL"),
    )
    default_generation_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_GENERATION_BASE_URL", "DEFAULT_GENERATION_BASE_URL"),
    )
    default_embeddings_provider: str = Field(
        default="openai_compatible",
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_EMBEDDINGS_PROVIDER", "DEFAULT_EMBEDDINGS_PROVIDER"),
    )
    default_embeddings_model: str = Field(
        default="text-embedding-005",
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_EMBEDDINGS_MODEL", "DEFAULT_EMBEDDINGS_MODEL"),
    )
    default_embeddings_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_EMBEDDINGS_BASE_URL", "DEFAULT_EMBEDDINGS_BASE_URL"),
    )
    default_reranker_provider: str = Field(
        default="cohere",
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_RERANKER_PROVIDER", "DEFAULT_RERANKER_PROVIDER"),
    )
    default_reranker_model: str = Field(
        default="rerank-v3.5",
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_RERANKER_MODEL", "DEFAULT_RERANKER_MODEL"),
    )
    default_reranker_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_RERANKER_BASE_URL", "DEFAULT_RERANKER_BASE_URL"),
    )
    default_mcp_endpoints: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_MCP_ENDPOINTS", "DEFAULT_MCP_ENDPOINTS"),
    )
    default_mcp_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_MCP_API_KEY", "DEFAULT_MCP_API_KEY"),
    )

    @field_validator("mcp_endpoints", mode="before")
    @classmethod
    def _split_endpoints(cls, value: Optional[str] | List[str]):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("default_mcp_endpoints", mode="before")
    @classmethod
    def _split_default_mcp_endpoints(cls, value: Optional[str] | List[str]):
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
