"""TerminalTool（终端只读/安全工具）：白名单指令、工作目录安全隔离与超时保护。

安全约束（对应审计报告 P0-1）：
1. 不使用 shell 执行：参数经 shlex 解析后交给 create_subprocess_exec，杜绝 ``&``、管道、
   重定向、命令替换等 shell 语义。
2. 命令首词必须在只读白名单内。
3. 任何会被解析为路径的参数，其 resolve() 结果必须位于 workspace_root 之内，
   防止 ``cat ../../etc/passwd`` 或绝对路径读取宿主机文件。
"""

from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path
from typing import Any

from app.observability.logging import get_logger
from app.tools.base import Tool, ToolContext, ToolResult

log = get_logger(__name__)

SAFE_COMMANDS = {"ls", "dir", "cat", "type", "head", "tail", "grep", "find", "wc", "echo"}

# 这些字符一旦出现即拒绝。执行层本身已不走 shell，这里作为纵深防御，
# 同时拦截审计中实测的 `echo hi & whoami` 形式。
DANGEROUS_TOKENS = {";", "&", "|", "`", "$", ">", "<", "\n", "\r"}


class TerminalTool(Tool):
    name = "terminal_tool"
    description = (
        "安全终端命令执行工具。仅支持只读和查看类命令（ls, dir, cat, head, tail, grep, echo）。"
        "禁止任何重定向、管道、系统修改或破坏性命令。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令字符串，例如 'ls' 或 'cat notes.txt'",
            }
        },
        "required": ["command"],
    }

    # 只读工具：即使未配置 API Token 也可通过 /tools、/mcp 调用。
    read_only = True

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _resolve_within(self, raw: str) -> Path | None:
        """把参数解析为 workspace_root 内的路径；越界或非法时返回 None。"""
        try:
            candidate = (self.workspace_root / raw).resolve()
        except (OSError, ValueError):
            return None
        try:
            if not candidate.is_relative_to(self.workspace_root):
                return None
        except (OSError, ValueError):
            return None
        return candidate

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        cmd_str = arguments.get("command", "").strip()
        log.info("terminal_tool_invoked", command=cmd_str, user_id=context.user_id)

        if not cmd_str:
            return ToolResult(ok=False, error="Command cannot be empty.")

        # 安全检查 1：禁止管道、重定向、命令替换等组合符号
        for token in DANGEROUS_TOKENS:
            if token in cmd_str:
                log.warning("terminal_command_rejected_token", token=token, command=cmd_str)
                return ToolResult(ok=False, error=f"Command contains prohibited token '{token}'.")

        # 解析命令（不做 shell 展开）
        try:
            parts = shlex.split(cmd_str, posix=os.name != "nt")
        except ValueError as e:
            return ToolResult(ok=False, error=f"Invalid command format: {e}")

        if not parts:
            return ToolResult(ok=False, error="Empty command arguments.")

        cmd_base = parts[0].lower()
        if cmd_base not in SAFE_COMMANDS:
            log.warning("terminal_command_rejected_whitelist", command=cmd_base)
            return ToolResult(ok=False, error=f"Command '{cmd_base}' is not in the allowed safe whitelist.")

        # 安全检查 2：任何路径型参数必须落在 workspace_root 内
        for arg in parts[1:]:
            if not arg or arg.startswith("-"):
                continue
            if self._resolve_within(arg) is None:
                log.warning("terminal_command_rejected_path_escape", argument=arg, command=cmd_str)
                return ToolResult(
                    ok=False, error=f"Path argument '{arg}' escapes the allowed workspace directory."
                )

        # 执行命令：直接 exec，不经 shell
        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_root),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            out_decoded = stdout.decode("utf-8", errors="replace")
            err_decoded = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                log.warning("terminal_command_failed", returncode=proc.returncode, stderr=err_decoded)
                return ToolResult(
                    ok=False, error=err_decoded or f"Exited with code {proc.returncode}", output=out_decoded
                )

            return ToolResult(ok=True, output=out_decoded)
        except TimeoutError:
            log.error("terminal_command_timeout", command=cmd_str)
            return ToolResult(ok=False, error="Command execution timed out (10s).")
        except FileNotFoundError:
            return ToolResult(ok=False, error=f"Command not found on this platform: '{parts[0]}'.")
        except Exception as e:  # noqa: BLE001
            log.error("terminal_command_exception", error=str(e))
            return ToolResult(ok=False, error=str(e))