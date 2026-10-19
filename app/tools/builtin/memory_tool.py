"""MemoryTool：允许 Agent 在推理循环中主动检索或存储用户记忆。"""

from __future__ import annotations

from typing import Any

from app.memory.manager import MemoryManager
from app.observability.logging import get_logger
from app.tools.base import Tool, ToolContext, ToolResult

log = get_logger(__name__)


class MemoryTool(Tool):
    name = "memory_tool"
    description = "用于主动向长期记忆系统写入关键信息，或按关键词主动检索历史记忆。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "save"],
                "description": "操作类型：search（搜索）或 save（保存）",
            },
            "query": {"type": "string", "description": "搜索关键词（action=search 时使用）"},
            "content": {"type": "string", "description": "要保存的记忆内容（action=save 时使用）"},
            "importance": {"type": "number", "description": "重要性评分（0.0 到 1.0，默认 0.7）"},
        },
        "required": ["action"],
    }

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action")
        user_id = context.user_id
        log.info("memory_tool_invoked", action=action, user_id=user_id)

        if action == "search":
            q = arguments.get("query", "")
            if not q:
                return ToolResult(ok=False, error="query cannot be empty for search")
            results = await self.memory.retrieve_all([q], user_id=user_id, top_k=5)
            if not results:
                return ToolResult(ok=True, output="未检索到相关记忆。")
            lines = [
                f"- [{sm.item.type.upper()}] (相关度: {sm.score:.2f}) {sm.item.content}" for sm in results
            ]
            return ToolResult(ok=True, output="\n".join(lines))

        elif action == "save":
            content = arguments.get("content", "")
            if not content:
                return ToolResult(ok=False, error="content cannot be empty for save")
            importance = float(arguments.get("importance", 0.7))
            mem_id = await self.memory.add(
                type="episodic",
                content=content,
                user_id=user_id,
                session_id=context.session_id,
                importance=importance,
            )
            return ToolResult(ok=True, output=f"记忆已成功存入情景记忆库（ID: {mem_id}）。")

        return ToolResult(ok=False, error=f"未知操作: {action}")
