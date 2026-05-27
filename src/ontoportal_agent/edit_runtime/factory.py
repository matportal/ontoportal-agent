from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..opencode_executor import OpenCodeAccountAuth, OpenCodeProviderAuth
from .base import EditRuntime
from .opencode import OpenCodeEditRuntime

_OPENCODE_ALIASES = {"", "opencode", "open-code", "workspace"}
_PI_ALIASES = {"pi", "pi.dev", "pi-dev"}


def normalize_edit_runtime_name(value: str | None) -> str:
    clean = str(value or "").strip().lower()
    if clean in _OPENCODE_ALIASES:
        return "opencode"
    if clean in _PI_ALIASES:
        return "pi"
    raise ValueError(f"Unsupported assistant edit runtime: {value}")


def create_edit_runtime(
    runtime_name: str | None = None,
    *,
    provider_auth: OpenCodeProviderAuth | None = None,
    account_auth: OpenCodeAccountAuth | None = None,
    mcp_servers: list[dict[str, Any] | str] | None = None,
) -> EditRuntime:
    """Create the configured edit runtime adapter.

    OpenCode remains the only executable runtime in this compatibility slice.
    Pi is intentionally recognized but unavailable until the adapter service,
    auth mapping, MCP/tool bridge, and safety validation are implemented.
    """

    settings = get_settings()
    runtime = normalize_edit_runtime_name(runtime_name or settings.edit_runtime_default)
    if runtime == "opencode":
        return OpenCodeEditRuntime(
            provider_auth=provider_auth,
            account_auth=account_auth,
            mcp_servers=mcp_servers,
        )
    raise ValueError("Pi edit runtime is not available until ONTOAGENT_EDIT_RUNTIME_DEFAULT=pi is backed by an adapter.")
