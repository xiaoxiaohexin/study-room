"""审计 P1-5 回归：题库按用户隔离，禁止跨用户取题与判分。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.learning.store import LearningStore
from app.storage.graph import NullGraphStore
from app.storage.sqlite import SQLiteStore


async def test_questions_are_scoped_per_user(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "q.db")
    await sqlite.initialize()
    try:
        store = LearningStore(sqlite, graph_store=NullGraphStore())
        kp = await store.upsert_knowledge_point("c1", "极限", "趋近过程", user_id="alice")
        q_alice = await store.add_question(
            kp_id=kp["id"], q_type="choice", stem="alice-q", answer="A", user_id="alice"
        )
        q_bob = await store.add_question(
            kp_id=kp["id"], q_type="choice", stem="bob-q", answer="A", user_id="bob"
        )

        # 按 ID 跨用户取题必须拿不到
        assert await store.get_question(q_alice["id"], user_id="bob") is None
        assert (await store.get_question(q_alice["id"], user_id="alice")) is not None

        # 待答题只返回自己的题
        pending_bob = await store.get_latest_unanswered_question("bob")
        assert pending_bob is not None and pending_bob["id"] == q_bob["id"]

        # 用他人的题判分必须被拒绝，且不产生任何记录
        with pytest.raises(ValueError):
            await store.record_attempt_and_update_mastery(
                user_id="bob", question_id=q_alice["id"], user_answer="A", is_correct=True, score=1.0
            )
    finally:
        await sqlite.aclose()