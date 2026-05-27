from __future__ import annotations

from typing import Any, Iterator

from ..opencode_executor import OpenCodeAccountAuth, OpenCodeExecutor, OpenCodeProviderAuth
from .base import EditRuntimeCapabilities, EditRuntimeRequest


class OpenCodeEditRuntime:
    """Edit runtime adapter that preserves the existing OpenCode behavior."""

    capabilities = EditRuntimeCapabilities(
        runtime="opencode",
        supports_sessions=True,
        supports_cancel=False,
        supports_mcp=True,
        supports_artifacts=True,
    )

    def __init__(
        self,
        *,
        provider_auth: OpenCodeProviderAuth | None = None,
        account_auth: OpenCodeAccountAuth | None = None,
        mcp_servers: list[dict[str, Any] | str] | None = None,
    ) -> None:
        self._executor = OpenCodeExecutor(
            provider_auth=provider_auth,
            account_auth=account_auth,
            mcp_servers=mcp_servers,
        )

    def stream(self, request: EditRuntimeRequest) -> Iterator[dict[str, Any]]:
        return self._executor.stream(
            prompt=request.prompt,
            thread_id=request.thread_id,
            trace_id=request.trace_id,
            resume_workspace=request.resume_workspace,
            resume_session_id=request.resume_session_id,
            task=request.task,
            retrieved_context=request.retrieved_context,
            citation_labels=list(request.citation_labels),
        )
