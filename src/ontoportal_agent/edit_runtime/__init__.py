"""Runtime-neutral edit workspace adapters for the assistant."""

from .base import EditRuntimeCapabilities, EditRuntimeRequest
from .deepagents import DeepAgentsEditRuntime
from .factory import create_edit_runtime, normalize_edit_runtime_name
from .opencode import OpenCodeEditRuntime

__all__ = [
    "EditRuntimeCapabilities",
    "EditRuntimeRequest",
    "DeepAgentsEditRuntime",
    "OpenCodeEditRuntime",
    "create_edit_runtime",
    "normalize_edit_runtime_name",
]
