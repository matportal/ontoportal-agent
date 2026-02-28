from __future__ import annotations

from dataclasses import dataclass
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
    def __init__(self, endpoints: List[str], api_key: str | None = None):
        self.endpoints = [endpoint.rstrip("/") for endpoint in endpoints if endpoint]
        self.api_key = api_key
        self._tool_cache: Optional[List[tuple[str, McpToolSpec]]] = None

    def _request_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def list_tools(self) -> List[tuple[str, McpToolSpec]]:
        if self._tool_cache is None:
            tools: List[tuple[str, McpToolSpec]] = []
            for endpoint in self.endpoints:
                url = f"{endpoint}/tools"
                try:
                    response = requests.get(url, headers=self._request_headers(), timeout=30)
                    response.raise_for_status()
                    payload = response.json()
                except requests.RequestException as err:
                    raise McpInvocationError(f"Failed to list tools from '{endpoint}': {err}") from err
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
                try:
                    response = requests.post(
                        tool.endpoint,
                        json={"tool": tool_name, "arguments": arguments},
                        headers=self._request_headers(),
                        timeout=120,
                    )
                except requests.RequestException as err:
                    raise McpInvocationError(f"Failed to invoke '{tool_name}' on '{endpoint}': {err}") from err
                if response.status_code >= 400:
                    raise McpInvocationError(response.text)
                return response.json()
        raise McpInvocationError(f"Tool '{tool_name}' not found in MCP endpoints")

    def invoke_rag_query(self, question: str, tool_name: str = "rag_query") -> Dict[str, Any]:
        return self.invoke(tool_name, {"query": question})
