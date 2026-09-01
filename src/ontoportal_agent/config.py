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
        default="gemini-3.1-pro-preview",
        validation_alias=AliasChoices("ONTOAGENT_LLM_MODEL", "LLM_MODEL"),
    )
    vertex_project: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_VERTEX_PROJECT", "VERTEX_PROJECT"),
    )
    vertex_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("ONTOAGENT_VERTEX_LOCATION", "VERTEX_LOCATION"),
    )
    vertex_service_account_json: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_VERTEX_SERVICE_ACCOUNT_JSON", "VERTEX_SERVICE_ACCOUNT_JSON"),
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
    opencode_path: str = Field(
        default="opencode",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_PATH", "OPENCODE_PATH"),
    )
    opencode_model: str = Field(
        default="opencode/big-pickle",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MODEL", "OPENCODE_MODEL"),
    )
    opencode_workspace_subdir: str = Field(
        default="opencode-runs",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_WORKSPACE_SUBDIR", "OPENCODE_WORKSPACE_SUBDIR"),
    )
    opencode_mcp_mode: str = Field(
        default="remote",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MCP_MODE", "OPENCODE_MCP_MODE"),
    )
    opencode_mcp_url: str = Field(
        default="https://mcp.matportal.org/mcp",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MCP_URL", "OPENCODE_MCP_URL"),
    )
    opencode_mcp_name: str = Field(
        default="ontoportal_api",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MCP_NAME", "OPENCODE_MCP_NAME"),
    )
    opencode_mcp_python: str = Field(
        default="python",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MCP_PYTHON", "OPENCODE_MCP_PYTHON"),
    )
    opencode_mcp_server_root: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MCP_SERVER_ROOT", "OPENCODE_MCP_SERVER_ROOT"),
    )
    opencode_mcp_transport: str = Field(
        default="stdio",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MCP_TRANSPORT", "OPENCODE_MCP_TRANSPORT"),
    )
    opencode_mcp_timeout_ms: int = Field(
        default=20000,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MCP_TIMEOUT_MS", "OPENCODE_MCP_TIMEOUT_MS"),
    )
    opencode_keep_workspace: bool = Field(
        default=True,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_KEEP_WORKSPACE", "OPENCODE_KEEP_WORKSPACE"),
    )
    opencode_max_log_lines: int = Field(
        default=400,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MAX_LOG_LINES", "OPENCODE_MAX_LOG_LINES"),
    )
    opencode_max_diff_chars: int = Field(
        default=24000,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_MAX_DIFF_CHARS", "OPENCODE_MAX_DIFF_CHARS"),
    )
    opencode_run_timeout_seconds: int = Field(
        default=900,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_RUN_TIMEOUT_SECONDS", "OPENCODE_RUN_TIMEOUT_SECONDS"),
    )
    opencode_artifact_retention_days: int = Field(
        default=7,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_ARTIFACT_RETENTION_DAYS",
            "OPENCODE_ARTIFACT_RETENTION_DAYS",
        ),
    )
    opencode_hybrid_ask_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_HYBRID_ASK_ENABLED", "OPENCODE_HYBRID_ASK_ENABLED"),
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
        default="gemini-3.1-pro-preview",
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

    @field_validator("opencode_mcp_mode")
    @classmethod
    def _normalize_opencode_mcp_mode(cls, value: str):
        normalized = str(value or "remote").strip().lower()
        if normalized not in {"remote", "local"}:
            raise ValueError("opencode_mcp_mode must be 'remote' or 'local'")
        return normalized

    def resolved_mcp_endpoints(self) -> List[str]:
        if self.mcp_endpoints:
            return self.mcp_endpoints
        return [self.rag_base_url.rstrip("/") + "/mcp"]


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    return AgentSettings()
