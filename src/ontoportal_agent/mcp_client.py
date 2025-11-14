from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests


@dataclass
class McpToolSpec:
    name: str
    description: str
    arguments: Dict[str, Any]
    endpoint: str


class McpInvocationError(RuntimeError):
    pass


class McpClient:
    def __init__(self, endpoints: List[str]):
        self.endpoints = [endpoint.rstrip("/") for endpoint in endpoints if endpoint]
        self._tool_cache: Optional[List[tuple[str, McpToolSpec]]] = None

    def list_tools(self) -> List[tuple[str, McpToolSpec]]:
        if self._tool_cache is None:
            tools: List[tuple[str, McpToolSpec]] = []
            for endpoint in self.endpoints:
                url = f"{endpoint}/tools"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                for tool in payload.get("tools", []):
                    tools.append(
                        (
                            endpoint,
                            McpToolSpec(
                                name=tool["name"],
                                description=tool.get("description", ""),
                                arguments=tool.get("arguments", {}),
                                endpoint=f"{endpoint}/invoke",
                            ),
                        )
                    )
            self._tool_cache = tools
        return list(self._tool_cache)

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        for endpoint, tool in self.list_tools():
            if tool.name == tool_name:
                response = requests.post(
                    tool.endpoint,
                    json={"tool": tool_name, "arguments": arguments},
                    timeout=120,
                )
                if response.status_code >= 400:
                    raise McpInvocationError(response.text)
                return response.json()
        raise McpInvocationError(f"Tool '{tool_name}' not found in MCP endpoints")
*** End of File
