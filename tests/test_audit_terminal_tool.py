"""审计 P0-1 回归：TerminalTool 不再经 shell 执行，且默认不注册。"""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.container import AppContainer
from app.tools.base import ToolContext
from app.tools.builtin.terminal_tool import TerminalTool


def _ctx(user_id: str = "u1") -> ToolContext:
    return ToolContext(user_id=user_id)


async def test_terminal_tool_rejects_shell_composition(tmp_path: Path):
    tool = TerminalTool(workspace_root=tmp_path)
    res = await tool.run({"command": "echo hi & whoami"}, _ctx())
    assert res.ok is False
    assert "prohibited token" in (res.error or "")


async def test_terminal_tool_rejects_path_escape(tmp_path: Path):
    (tmp_path.parent / "outside_secret.txt").write_text("secret", encoding="utf-8")
    tool = TerminalTool(workspace_root=tmp_path)
    res = await tool.run({"command": "cat ../outside_secret.txt"}, _ctx())
    assert res.ok is False
    assert "escapes" in (res.error or "")


async def test_terminal_tool_runs_whitelisted_command_without_shell(tmp_path: Path):
    tool = TerminalTool(workspace_root=tmp_path)
    res = await tool.run({"command": "echo hello"}, _ctx())
    assert res.ok is True
    assert "hello" in res.output


async def test_terminal_tool_not_registered_by_default(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "c.db",
        data_dir=tmp_path / "data",
        enable_terminal_tool=False,
    )
    container = AppContainer(settings)
    await container.initialize()
    try:
        assert container.tool_registry.get("terminal_tool") is None
        assert container.status.get("terminal_tool") == "disabled"
    finally:
        await container.aclose()

    enabled = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "c2.db",
        data_dir=tmp_path / "data2",
        enable_terminal_tool=True,
    )
    container2 = AppContainer(enabled)
    await container2.initialize()
    try:
        assert container2.tool_registry.get("terminal_tool") is not None
    finally:
        await container2.aclose()