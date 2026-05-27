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
    def __init__(self, endpoints: List[str] | List[dict[str, Any]], api_key: str | None = None):
        self.endpoint_configs: List[dict[str, Any]] = []
        for endpoint in endpoints:
            if isinstance(endpoint, dict):
                url = (endpoint.get("url") or "").rstrip("/")
                per_endpoint_api_key = endpoint.get("api_key")
                raw_headers = endpoint.get("headers")
                timeout_ms = endpoint.get("timeout_ms")
                if not url:
                    continue
                try:
                    timeout_ms = int(timeout_ms)
                except (TypeError, ValueError):
                    timeout_ms = 0
                headers: Dict[str, str] = {}
                if isinstance(raw_headers, dict):
                    for key, value in raw_headers.items():
                        header_name = str(key or "").strip()
                        if not header_name:
                            continue
                        header_value = str(value or "").strip()
                        if not header_value:
                            continue
                        headers[header_name] = header_value
                self.endpoint_configs.append(
                    {
                        "url": url,
                        "api_key": per_endpoint_api_key,
                        "headers": headers,
                        "timeout_ms": max(0, timeout_ms),
                    }
                )
                continue

            url = str(endpoint).rstrip("/")
            if not url:
                continue
            self.endpoint_configs.append({"url": url, "api_key": None, "headers": {}, "timeout_ms": 0})

        self.endpoints = [cfg["url"] for cfg in self.endpoint_configs]
        self.api_key = api_key
        self._tool_cache: Optional[List[tuple[str, McpToolSpec]]] = None

    def _endpoint_config(self, endpoint: str | None) -> dict[str, Any]:
        if not endpoint:
            return {}
        for cfg in self.endpoint_configs:
            if cfg["url"] == endpoint:
                return cfg
        return {}

    def _timeout_seconds(self, endpoint: str | None, default_seconds: int) -> int:
        cfg = self._endpoint_config(endpoint)
        timeout_ms = cfg.get("timeout_ms")
        try:
            timeout_ms = int(timeout_ms)
        except (TypeError, ValueError):
            timeout_ms = 0
        if timeout_ms <= 0:
            return default_seconds
        timeout_seconds = max(1, timeout_ms // 1000)
        return timeout_seconds

    def _request_headers(self, endpoint: str | None = None) -> Dict[str, str]:
        endpoint_config = self._endpoint_config(endpoint)
        headers: Dict[str, str] = dict(endpoint_config.get("headers") or {})
        endpoint_api_key = endpoint_config.get("api_key")
        resolved_api_key = endpoint_api_key or self.api_key
        has_api_key_header = any(str(key).lower() == "x-api-key" for key in headers)
        if resolved_api_key and not has_api_key_header:
            headers["X-API-Key"] = str(resolved_api_key)
        return headers

    def list_tools(self) -> List[tuple[str, McpToolSpec]]:
        if self._tool_cache is None:
            tools: List[tuple[str, McpToolSpec]] = []
            errors: List[str] = []
            for endpoint in self.endpoints:
                url = f"{endpoint}/tools"
                try:
                    response = requests.get(
                        url,
                        headers=self._request_headers(endpoint),
                        timeout=self._timeout_seconds(endpoint, default_seconds=30),
                    )
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, RuntimeError, ValueError, TypeError, KeyError) as err:
                    errors.append(f"{endpoint}: {err}")
                    continue
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
            if not tools and errors:
                raise McpInvocationError(f"Failed to list tools from configured MCP endpoints: {'; '.join(errors)}")
            self._tool_cache = tools
        return list(self._tool_cache)

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        for endpoint, tool in self.list_tools():
            if tool.name == tool_name:
                try:
                    response = requests.post(
                        tool.endpoint,
                        json={"tool": tool_name, "arguments": arguments},
                        headers=self._request_headers(endpoint),
                        timeout=self._timeout_seconds(endpoint, default_seconds=120),
                    )
                except requests.RequestException as err:
                    raise McpInvocationError(f"Failed to invoke '{tool_name}' on '{endpoint}': {err}") from err
                if response.status_code >= 400:
                    raise McpInvocationError(response.text)
                return response.json()
        raise McpInvocationError(f"Tool '{tool_name}' not found in MCP endpoints")

    def invoke_rag_query(
        self,
        question: str,
        tool_name: str = "rag_query",
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {"query": question}
        if top_k is not None:
            arguments["top_k"] = int(top_k)
        return self.invoke(tool_name, arguments)
