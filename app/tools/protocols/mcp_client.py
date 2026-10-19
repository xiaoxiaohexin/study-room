"""多智能体与外部协议协议适配层（第 10 章 & 14 章）：MCP / A2A / ANP 协议桩与适配器。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.observability.logging import get_logger
from app.tools.base import Tool, ToolContext, ToolResult

log = get_logger(__name__)


class McpToolAdapter(Tool):
    """MCP 协议工具适配器桩：将 FastMCP / MCP Server 暴露的工具转换为统一 Tool 接口。"""

    def __init__(self, name: str, description: str, parameters_schema: dict[str, Any], server_uri: str):
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self.server_uri = server_uri

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        log.info("mcp_tool_forwarding", tool_name=self.name, server_uri=self.server_uri)
        # 实际通过 SSE/Stdio 转发请求给 MCP 服务
        return ToolResult(ok=True, output=f"[MCP-Mock] Tool '{self.name}' executed on {self.server_uri}")


class A2ATask(BaseModel):
    """A2A 智能体点对点任务委托协议模型。"""

    task_id: str
    from_agent: str
    to_agent: str
    instruction: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class AnpIdentity(BaseModel):
    """ANP (Agent Network Protocol) 去中心化注册中心与 DID 寻址模型。"""

    agent_did: str
    capabilities: list[str] = Field(default_factory=list)
    endpoint: str
