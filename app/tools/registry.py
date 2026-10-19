"""工具注册中心（ToolRegistry）：统一注册、元数据导出与安全隔离执行。"""

from __future__ import annotations

import asyncio
import json
import time

from app.observability.logging import get_logger
from app.schemas.messages import ToolSpec
from app.tools.base import Tool, ToolContext, ToolResult

log = get_logger(__name__)


class ToolRegistry:
    def __init__(self, default_timeout: float = 30.0, max_output_chars: int = 4000):
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        log.info("tool_registered", tool_name=tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        if names:
            selected = [self._tools[n].get_spec() for n in names if n in self._tools]
            if selected:
                return selected
        return [tool.get_spec() for tool in self._tools.values()]

    def describe_for_prompt(self, names: list[str] | None = None) -> str:
        """为文本协议提示词生成工具描述（支持按模式白名单过滤）。"""
        tools = [self._tools[n] for n in names if n in self._tools] if names else []
        if not tools:
            tools = list(self._tools.values())
        lines = []
        for tool in tools:
            schema_str = json.dumps(tool.parameters_schema, ensure_ascii=False)
            lines.append(f"- {tool.name}: {tool.description}\n  参数模式: {schema_str}")
        return "\n".join(lines)

    async def execute(
        self,
        name: str,
        arguments: dict,
        context: ToolContext,
        timeout: float | None = None,
    ) -> ToolResult:
        tool = self.get(name)
        if not tool:
            log.warning("tool_not_found", tool_name=name)
            return ToolResult(ok=False, error=f"Tool '{name}' not found.")

        t_limit = timeout or self.default_timeout
        start_time = time.perf_counter()
        log.info("tool_execution_started", tool_name=name, user_id=context.user_id, timeout=t_limit)

        try:
            res: ToolResult = await asyncio.wait_for(tool.run(arguments, context), timeout=t_limit)
        except TimeoutError:
            log.error("tool_execution_timeout", tool_name=name, timeout=t_limit)
            return ToolResult(ok=False, error=f"Tool '{name}' execution timed out after {t_limit}s.")
        except Exception as e:  # noqa: BLE001
            log.error("tool_execution_exception", tool_name=name, error=str(e))
            return ToolResult(ok=False, error=f"Tool '{name}' failed with error: {e}")

        # 输出长度截断保护
        if len(res.output) > self.max_output_chars:
            res.output = (
                res.output[: self.max_output_chars]
                + f"\n... [Output truncated to {self.max_output_chars} chars]"
            )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log.info(
            "tool_execution_completed",
            tool_name=name,
            ok=res.ok,
            duration_ms=elapsed_ms,
            output_len=len(res.output),
        )
        return res
