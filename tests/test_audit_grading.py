"""审计 P0 回归：主观题判分兜底不得臆断正确，必须返回 ungraded 且不写 SM-2。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.learning.store import LearningStore
from app.storage.graph import NullGraphStore
from app.storage.sqlite import SQLiteStore
from app.tools.base import ToolContext
from app.tools.builtin.learning_tools import GradeTool


class _RaisingLLM:
    """LLM 结构化判分不可用时的替身。"""

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise RuntimeError("llm offline")

    async def aclose(self) -> None:
        return None


async def test_subjective_grade_fallback_is_ungraded(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "grade.db")
    await sqlite.initialize()
    try:
        store = LearningStore(sqlite, graph_store=NullGraphStore())
        kp = await store.upsert_knowledge_point(
            course_id="c1", name="导数", summary="变化率", user_id="alice"
        )
        q = await store.add_question(
            kp_id=kp["id"],
            course_id="c1",
            q_type="short_answer",
            stem="解释导数的几何意义",
            answer="切线斜率",
            user_id="alice",
        )

        tool = GradeTool(store=store, llm=_RaisingLLM())
        res = await tool.run(
            {"user_answer": "这是一个足够长的错误回答，长度超过十个字", "question_id": q["id"]},
            ToolContext(user_id="alice"),
        )
        assert res.ok is True
        assert res.metadata.get("status") == "ungraded"

        cur = await sqlite.conn.execute(
            "SELECT COUNT(*) AS n FROM attempts WHERE question_id = ?", (q["id"],)
        )
        row = await cur.fetchone()
        assert row["n"] == 0

        # 初始掌握度由知识点创建时写入，判分兜底不得改写它，也不得生成复习排期。
        cur = await sqlite.conn.execute(
            "SELECT mastery, repetitions FROM mastery WHERE user_id = ? AND kp_id = ?",
            ("alice", kp["id"]),
        )
        row = await cur.fetchone()
        assert row is not None
        assert float(row["mastery"]) == pytest.approx(0.3)
        assert int(row["repetitions"]) == 0
    finally:
        await sqlite.aclose()