from __future__ import annotations

from typing import Any

from ..agent.options import AgentRuntimeOptions
from ..config import get_settings
from ..opencode_executor import OpenCodeAccountAuth, OpenCodeProviderAuth
from .base import EditRuntime
from .deepagents import DeepAgentsEditRuntime
from .opencode import OpenCodeEditRuntime

_OPENCODE_ALIASES = {"", "opencode", "open-code", "workspace"}
_DEEPAGENTS_ALIASES = {"deepagents", "deep-agents", "langchain-deepagents", "langchain_deepagents"}
_PI_ALIASES = {"pi", "pi.dev", "pi-dev"}


def normalize_edit_runtime_name(value: str | None) -> str:
    clean = str(value or "").strip().lower()
    if clean in _OPENCODE_ALIASES:
        return "opencode"
    if clean in _DEEPAGENTS_ALIASES:
        return "deepagents"
    if clean in _PI_ALIASES:
        return "pi"
    raise ValueError(f"Unsupported assistant edit runtime: {value}")


def create_edit_runtime(
    runtime_name: str | None = None,
    *,
    provider_auth: OpenCodeProviderAuth | None = None,
    account_auth: OpenCodeAccountAuth | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    runtime_options: AgentRuntimeOptions | None = None,
    model: Any | None = None,
) -> EditRuntime:
    """Create the configured edit runtime adapter.

    OpenCode remains the default runtime. Deep Agents is executable only when
    explicitly enabled. Pi is intentionally recognized but unavailable until a
    separate adapter service, auth mapping, MCP/tool bridge, and safety
    validation are implemented.
    """

    settings = get_settings()
    runtime = normalize_edit_runtime_name(runtime_name or settings.edit_runtime_default)
    if runtime == "opencode":
        return OpenCodeEditRuntime(
            provider_auth=provider_auth,
            account_auth=account_auth,
            mcp_servers=mcp_servers,
        )
    if runtime == "deepagents":
        if not settings.deepagents_enabled:
            raise ValueError("Deep Agents edit runtime is disabled by ONTOAGENT_DEEPAGENTS_ENABLED.")
        if account_auth is not None and str(account_auth.kind or "").strip().lower() != "gemini_antigravity":
            # Codex subscription auth remains OpenCode-specific today. Preserve those
            # users by routing to the mature OpenCode runtime until Deep Agents has a
            # native Codex account provider.
            return OpenCodeEditRuntime(
                provider_auth=provider_auth,
                account_auth=account_auth,
                mcp_servers=mcp_servers,
            )
        return DeepAgentsEditRuntime(
            runtime_options=runtime_options,
            mcp_servers=mcp_servers,
            account_auth=account_auth,
            model=model,
        )
    raise ValueError("Pi edit runtime is not available until ONTOAGENT_PI_ADAPTER_ENABLED is backed by an adapter.")
