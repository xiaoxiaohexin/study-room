"""审计 P1-3 回归：NoteTool 必须拒绝路径穿越。"""

from __future__ import annotations

from pathlib import Path

from app.storage.sqlite import SQLiteStore
from app.tools.base import ToolContext
from app.tools.builtin.note_tool import NoteTool


async def test_note_tool_rejects_path_traversal(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "notes.db")
    await sqlite.initialize()
    try:
        tool = NoteTool(notes_root=tmp_path / "notes", sqlite=sqlite)

        res = await tool.run(
            {"action": "read", "note_id": "../../outside/secret"}, ToolContext(user_id="alice")
        )
        assert res.ok is False

        res_create = await tool.run(
            {"action": "create", "title": "x", "content": "y"}, ToolContext(user_id="../../evil")
        )
        assert res_create.ok is False
        assert not (tmp_path / "evil").exists()
    finally:
        await sqlite.aclose()