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

    # Workspace and edit runtime
    ontology_workdir: Path = Field(
        default=Path("/tmp/ontoportal-agent"),
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_WORKDIR", "ONTOLOGY_WORKDIR"),
    )
    edit_runtime_default: str = Field(
        default="opencode",
        validation_alias=AliasChoices("ONTOAGENT_EDIT_RUNTIME_DEFAULT", "EDIT_RUNTIME_DEFAULT"),
    )
    edit_runtime_dual_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_EDIT_RUNTIME_DUAL_ENABLED", "EDIT_RUNTIME_DUAL_ENABLED"),
    )
    deepagents_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_DEEPAGENTS_ENABLED", "DEEPAGENTS_ENABLED"),
    )
    deepagents_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_DEEPAGENTS_MODEL", "DEEPAGENTS_MODEL"),
    )
    deepagents_antigravity_base_url: str = Field(
        default="http://localhost:51200/v1",
        validation_alias=AliasChoices("ONTOAGENT_DEEPAGENTS_ANTIGRAVITY_BASE_URL", "DEEPAGENTS_ANTIGRAVITY_BASE_URL"),
    )
    deepagents_antigravity_api_key: str = Field(
        default="proxy-managed",
        validation_alias=AliasChoices("ONTOAGENT_DEEPAGENTS_ANTIGRAVITY_API_KEY", "DEEPAGENTS_ANTIGRAVITY_API_KEY"),
    )
    pi_path: str = Field(
        default="pi",
        validation_alias=AliasChoices("ONTOAGENT_PI_PATH", "PI_PATH"),
    )
    pi_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_PI_MODEL", "PI_MODEL"),
    )
    pi_session_subdir: str = Field(
        default=".pi-sessions",
        validation_alias=AliasChoices("ONTOAGENT_PI_SESSION_SUBDIR", "PI_SESSION_SUBDIR"),
    )
    pi_adapter_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_PI_ADAPTER_ENABLED", "PI_ADAPTER_ENABLED"),
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
    opencode_rag_mcp_name: str = Field(
        default="matportal_rag",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_RAG_MCP_NAME", "OPENCODE_RAG_MCP_NAME"),
    )
    opencode_rag_mcp_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_RAG_MCP_URL", "OPENCODE_RAG_MCP_URL"),
    )
    opencode_rag_mcp_timeout_ms: int = Field(
        default=30000,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_RAG_MCP_TIMEOUT_MS", "OPENCODE_RAG_MCP_TIMEOUT_MS"),
    )
    opencode_antigravity_plugin: str = Field(
        default="opencode-antigravity-auth@latest",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_ANTIGRAVITY_PLUGIN", "OPENCODE_ANTIGRAVITY_PLUGIN"),
    )
    opencode_antigravity_model: str = Field(
        default="google/antigravity-gemini-3-pro",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_ANTIGRAVITY_MODEL", "OPENCODE_ANTIGRAVITY_MODEL"),
    )
    opencode_exa_websearch_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_EXA_WEBSEARCH_ENABLED", "OPENCODE_EXA_WEBSEARCH_ENABLED"),
    )
    opencode_robot_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_ROBOT_ENABLED", "OPENCODE_ROBOT_ENABLED"),
    )
    opencode_robot_jar_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_ROBOT_JAR_PATH", "OPENCODE_ROBOT_JAR_PATH"),
    )
    opencode_robot_java_path: str = Field(
        default="java",
        validation_alias=AliasChoices("ONTOAGENT_OPENCODE_ROBOT_JAVA_PATH", "OPENCODE_ROBOT_JAVA_PATH"),
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
    opencode_block_dangerous_commands: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_BLOCK_DANGEROUS_COMMANDS",
            "OPENCODE_BLOCK_DANGEROUS_COMMANDS",
        ),
    )
    opencode_interactive_sessions_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_INTERACTIVE_SESSIONS_ENABLED",
            "OPENCODE_INTERACTIVE_SESSIONS_ENABLED",
        ),
    )
    opencode_strict_workflow_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_STRICT_WORKFLOW_ENABLED",
            "OPENCODE_STRICT_WORKFLOW_ENABLED",
        ),
    )
    opencode_apply_publish_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_APPLY_PUBLISH_ENABLED",
            "OPENCODE_APPLY_PUBLISH_ENABLED",
        ),
    )
    opencode_mobi_handoff_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_MOBI_HANDOFF_ENABLED",
            "OPENCODE_MOBI_HANDOFF_ENABLED",
        ),
    )
    opencode_strict_sandbox_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_STRICT_SANDBOX_ENABLED",
            "OPENCODE_STRICT_SANDBOX_ENABLED",
        ),
    )
    opencode_global_concurrency_limit: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_GLOBAL_CONCURRENCY_LIMIT",
            "OPENCODE_GLOBAL_CONCURRENCY_LIMIT",
        ),
    )
    opencode_user_concurrency_limit: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "ONTOAGENT_OPENCODE_USER_CONCURRENCY_LIMIT",
            "OPENCODE_USER_CONCURRENCY_LIMIT",
        ),
    )

    # Ontology copilot feature gates. Keep all risky behavior default-off.
    ontology_copilot_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_COPILOT_ENABLED", "ONTOLOGY_COPILOT_ENABLED"),
    )
    ontology_ui_panels_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_UI_PANELS_ENABLED", "ONTOLOGY_UI_PANELS_ENABLED"),
    )
    ontology_method_panel_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_METHOD_PANEL_ENABLED", "ONTOLOGY_METHOD_PANEL_ENABLED"),
    )
    ontology_reuse_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_REUSE_ENABLED", "ONTOLOGY_REUSE_ENABLED"),
    )
    ontology_advanced_validation_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "ONTOAGENT_ONTOLOGY_ADVANCED_VALIDATION_ENABLED",
            "ONTOLOGY_ADVANCED_VALIDATION_ENABLED",
        ),
    )
    ontology_reasoner_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_REASONER_ENABLED", "ONTOLOGY_REASONER_ENABLED"),
    )
    ontology_shacl_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_SHACL_ENABLED", "ONTOLOGY_SHACL_ENABLED"),
    )
    ontology_build_profiles_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ONTOAGENT_ONTOLOGY_BUILD_PROFILES_ENABLED", "ONTOLOGY_BUILD_PROFILES_ENABLED"),
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
    default_mcp_auth_mode: str = Field(
        default="none",
        validation_alias=AliasChoices("ONTOAGENT_DEFAULT_MCP_AUTH_MODE", "DEFAULT_MCP_AUTH_MODE"),
    )
    mcp_bot_username: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_MCP_BOT_USERNAME", "MCP_BOT_USERNAME"),
    )
    mcp_bot_password: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ONTOAGENT_MCP_BOT_PASSWORD", "MCP_BOT_PASSWORD"),
    )

    @field_validator("opencode_global_concurrency_limit", "opencode_user_concurrency_limit")
    @classmethod
    def _positive_concurrency_limit(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("OpenCode concurrency limits must be at least 1")
        return int(value)

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

    @field_validator("default_mcp_auth_mode")
    @classmethod
    def _normalize_default_mcp_auth_mode(cls, value: str) -> str:
        normalized = str(value or "none").strip().lower()
        if normalized not in {"none", "api_key", "basic_user", "basic_bot"}:
            raise ValueError("default_mcp_auth_mode must be one of: none, api_key, basic_user, basic_bot")
        return normalized

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
