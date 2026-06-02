from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass(frozen=True)
class EditRuntimeRequest:
    """Runtime-neutral request for an assistant edit workspace run."""

    prompt: str
    thread_id: str | None
    trace_id: str
    resume_workspace: str | None = None
    resume_session_id: str | None = None
    task: str = "edit"
    retrieved_context: str = ""
    citation_labels: tuple[str, ...] = field(default_factory=tuple)
    context: dict[str, Any] | None = None


@dataclass(frozen=True)
class EditRuntimeCapabilities:
    """Describes runtime support that rollout code can gate on."""

    runtime: str
    supports_sessions: bool = False
    supports_cancel: bool = False
    supports_mcp: bool = False
    supports_artifacts: bool = True


class EditRuntime(Protocol):
    """Adapter protocol for coding/edit runtimes.

    Implementations yield normalized assistant stream event dictionaries and return
    the runtime-specific result object as the generator return value.
    """

    capabilities: EditRuntimeCapabilities

    def stream(self, request: EditRuntimeRequest) -> Iterator[dict[str, Any]]:
        ...
