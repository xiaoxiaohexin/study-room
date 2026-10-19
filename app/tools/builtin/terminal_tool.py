"""TerminalTool（终端只读/安全工具）：白名单指令、工作目录安全隔离与超时保护。"""

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
DANGEROUS_TOKENS = {";", "&&", "||", "|", "`", "$", ">", "<", "\n", "\r"}


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

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        cmd_str = arguments.get("command", "").strip()
        log.info("terminal_tool_invoked", command=cmd_str, user_id=context.user_id)

        if not cmd_str:
            return ToolResult(ok=False, error="Command cannot be empty.")

        # 安全检查 1：禁止管道与重定向等组合符号
        for token in DANGEROUS_TOKENS:
            if token in cmd_str:
                log.warning("terminal_command_rejected_token", token=token, command=cmd_str)
                return ToolResult(ok=False, error=f"Command contains prohibited token '{token}'.")

        # 解析命令
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

        # 执行命令，固定工作目录为 workspace_root
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd_str,
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
        except Exception as e:  # noqa: BLE001
            log.error("terminal_command_exception", error=str(e))
            return ToolResult(ok=False, error=str(e))
