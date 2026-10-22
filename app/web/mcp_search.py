"""最小 MCP Streamable-HTTP 客户端：把外部 MCP 搜索服务当作联网搜索 Provider 使用。

只实现联网补充需要的子集：initialize -> notifications/initialized -> tools/list -> tools/call。
不引入额外 SDK 依赖，直接用 httpx 处理 JSON 或 SSE 响应。
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.core.config import Settings
from app.observability.logging import get_logger
from app.web.search import WebSearchError, WebSearchResult

log = get_logger(__name__)


def _parse_jsonrpc_payload(text: str) -> list[dict[str, Any]]:
    """解析 MCP 响应：支持纯 JSON、单条 SSE、多条 SSE data 帧。"""
    body = (text or "").strip()
    if not body:
        return []
    messages: list[dict[str, Any]] = []
    if any(line.strip().startswith("data:") for line in body.splitlines()):
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                messages.append(parsed)
        return messages
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [m for m in parsed if isinstance(m, dict)]
    return [parsed] if isinstance(parsed, dict) else []


def _extract_results_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if isinstance(result.get("text"), str):
        return result["text"]
    return json.dumps(result, ensure_ascii=False)


def _coerce_search_results(raw_text: str) -> list[WebSearchResult]:
    """容忍常见 MCP 搜索工具返回格式：JSON 数组/对象或纯文本行。"""
    text = (raw_text or "").strip()
    items: list[dict[str, Any]] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            items = [x for x in parsed if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            for key in ("results", "items", "data", "documents"):
                value = parsed.get(key)
                if isinstance(value, list):
                    items = [x for x in value if isinstance(x, dict)]
                    break
            if not items:
                items = [parsed]
    except json.JSONDecodeError:
        # 纯文本：按行解析 "标题 URL 摘要"
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.search(r"https?://\S+", line)
            if not match:
                continue
            url = re.sub(r"[)。，,.;]+$", "", match.group(0))
            title = line.replace(match.group(0), "").strip() or url
            items.append({"url": url, "title": title, "snippet": ""})

    results: list[WebSearchResult] = []
    for item in items:
        url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
        if not url:
            continue
        results.append(
            WebSearchResult(
                title=str(item.get("title") or item.get("name") or url)[:200],
                url=url,
                snippet=str(item.get("snippet") or item.get("content") or item.get("description") or "")[:500],
                provider="mcp",
            )
        )
    return results


class McpSearchClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._session_id: str | None = None
        self._next_id = 0

    @property
    def configured(self) -> bool:
        return bool(self.settings.mcp_web_server_url.strip())

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _build_payload(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        return {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}

    async def _post(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> list[dict[str, Any]]:
        resp = await client.post(
            self.settings.mcp_web_server_url,
            json=payload,
            headers=self._headers(),
        )
        session_id = resp.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        resp.raise_for_status()
        return _parse_jsonrpc_payload(resp.text)

    @staticmethod
    def _find_result(messages: list[dict[str, Any]], request_id: int) -> dict[str, Any] | None:
        for message in messages:
            if message.get("id") == request_id and "result" in message:
                return message["result"]
        return None

    async def _initialize(self, client: httpx.AsyncClient) -> None:
        payload = self._build_payload(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "xiaoxiao-studyroom", "version": "0.3"},
            },
        )
        await self._post(client, payload)
        # notifications/initialized 是通知（无 id），部分服务器返回 202
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            await self._post(client, notification)
        except httpx.HTTPError:
            pass

    async def _call_tool(self, client: httpx.AsyncClient, tool_name: str, arguments: dict[str, Any]) -> str:
        payload = self._build_payload("tools/call", {"name": tool_name, "arguments": arguments})
        messages = await self._post(client, payload)
        result = self._find_result(messages, payload["id"])
        if result is None:
            error = next((m.get("error") for m in messages if m.get("error")), None)
            raise WebSearchError(f"MCP tools/call 失败: {error}")
        return _extract_results_text(result)

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        if not self.configured:
            raise WebSearchError("未配置 MCP 搜索服务（MCP_WEB_SERVER_URL）")
        tool_name = self.settings.mcp_web_search_tool or "search"
        arguments = {"query": query, "max_results": max(1, top_k), "top_k": max(1, top_k)}
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.mcp_web_timeout,
                trust_env=False,
                follow_redirects=True,
            ) as client:
                await self._initialize(client)
                raw_text = await self._call_tool(client, tool_name, arguments)
        except httpx.HTTPError as exc:
            raise WebSearchError(f"MCP 搜索服务请求失败: {exc}") from exc

        results = _coerce_search_results(raw_text)
        log.info("mcp_search_completed", query=query, count=len(results), tool=tool_name)
        return results[: max(1, top_k)]
