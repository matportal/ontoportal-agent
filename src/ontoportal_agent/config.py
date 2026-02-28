from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Runtime configuration for the OntoPortal agent."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ONTOAGENT_", extra="ignore")

    # LLM settings
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(default=None, alias="OPENAI_API_BASE")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # RAG endpoint
    rag_base_url: str = Field(default="http://localhost:8000", alias="RAG_BASE_URL")
    rag_query_path: str = Field(default="/api/v1/query", alias="RAG_QUERY_PATH")

    # OntoPortal REST API
    ontoportal_api_base: str = Field(default="https://rest.matportal.org", alias="ONTOPORTAL_API_BASE")
    ontoportal_api_key: str = Field(..., alias="ONTOPORTAL_API_KEY")

    # Workspace
    ontology_workdir: Path = Field(default=Path("/tmp/ontoportal-agent"), alias="ONTOLOGY_WORKDIR")

    # Approval gate
    require_manual_approval: bool = Field(default=True, alias="REQUIRE_MANUAL_APPROVAL")

    # Model Context Protocol endpoints (comma-separated)
    mcp_endpoints: List[str] = Field(default_factory=list, alias="MCP_ENDPOINTS")
    mcp_api_key: Optional[str] = Field(default=None, alias="MCP_API_KEY")
    mcp_rag_tool_name: str = Field(default="rag_query", alias="MCP_RAG_TOOL_NAME")

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
