"""工具基类、上下文与执行结果定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.messages import ToolSpec


class ToolContext(BaseModel):
    user_id: str
    session_id: str | None = None
    workspace_dir: str = "data"
    # Session-scoped retrieval scope: tools must never reach outside these bindings.
    rag_namespace: str = "default"
    course_id: str | None = None


class ToolResult(BaseModel):
    ok: bool = True
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_str(self) -> str:
        if not self.ok:
            return f"Error: {self.error or 'Unknown error'}"
        return self.output


class Tool(ABC):
    name: str
    description: str
    parameters_schema: dict[str, Any]
    # 只读工具不改变任何服务端状态；未配置 API Token 时仅允许通过 /tools、/mcp
    # 调用只读工具，避免恶意网页借本机浏览器触发高风险操作。
    read_only: bool = False

    @abstractmethod
    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """执行工具逻辑。"""
        ...

    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )
