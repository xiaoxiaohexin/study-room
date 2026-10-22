"""Tests for SQLite session schema migration, SessionStateManager CRUD, and Session REST APIs.
Milestone M1 verification suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app
from app.storage.sqlite import SQLiteStore


@pytest.fixture
async def app_instance(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "test_session.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app
    await container.aclose()


@pytest.mark.asyncio
async def test_idempotent_sqlite_migration_legacy_db(tmp_path: Path):
    """验证向后兼容迁移：从无 course_id 与 rag_namespace 的旧版 SQLite 数据库启动时平滑迁移并保留老数据。"""
    legacy_db_path = tmp_path / "legacy.db"

    # 1. 模拟旧版数据表（缺少 course_id, rag_namespace）
    async with aiosqlite.connect(legacy_db_path) as db:
        await db.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        await db.execute(
            "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("legacy_sess_001", "user_legacy", "旧会话标题", 1600000000.0, 1600000000.0),
        )
        await db.commit()

    # 2. 第一次运行 SQLiteStore.initialize()
    store = SQLiteStore(legacy_db_path)
    await store.initialize()

    # 验证字段迁移成功
    cur = await store.conn.execute("PRAGMA table_info(sessions)")
    columns = {row["name"]: row for row in await cur.fetchall()}
    assert "course_id" in columns
    assert "rag_namespace" in columns

    # 验证已有数据的 rag_namespace 回填为 'default'，course_id 为 NULL
    cur = await store.conn.execute(
        "SELECT session_id, user_id, title, course_id, rag_namespace FROM sessions WHERE session_id = ?",
        ("legacy_sess_001",),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row["user_id"] == "user_legacy"
    assert row["title"] == "旧会话标题"
    assert row["course_id"] is None
    assert row["rag_namespace"] == "default"

    # 验证索引已成功创建
    cur_idx = await store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'sessions'"
    )
    idx_names = {r["name"] for r in await cur_idx.fetchall()}
    assert "idx_sessions_course" in idx_names
    assert "idx_sessions_rag_ns" in idx_names

    # 3. 第二次调用 initialize() 验证幂等性无报错
    await store.initialize()
    await store.aclose()


@pytest.mark.asyncio
async def test_create_session_api_endpoints(app_instance):
    """验证 POST /api/sessions 支持创建默认会话与自定义 course_id / rag_namespace 会话。"""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. 默认创建
        res_default = await ac.post("/api/sessions", json={"user_id": "student_1"})
        assert res_default.status_code == 200
        d1 = res_default.json()
        assert d1["user_id"] == "student_1"
        assert d1["session_id"]
        assert d1["title"] == "新会话"
        assert d1["course_id"] is None
        assert d1["rag_namespace"] == "default"
        assert isinstance(d1["created_at"], float)
        assert isinstance(d1["updated_at"], float)

        # 2. 自定义参数创建
        custom_payload = {
            "user_id": "student_1",
            "session_id": "sess_cs101_01",
            "title": "计算机网络第一章",
            "course_id": "course_cs101",
            "rag_namespace": "cs_network_v1",
        }
        res_custom = await ac.post("/api/sessions", json=custom_payload)
        assert res_custom.status_code == 200
        d2 = res_custom.json()
        assert d2["user_id"] == "student_1"
        assert d2["session_id"] == "sess_cs101_01"
        assert d2["title"] == "计算机网络第一章"
        assert d2["course_id"] == "course_cs101"
        assert d2["rag_namespace"] == "cs_network_v1"

        # 3. 对已有 session_id 再次调用 POST，传新参数时就地更新
        update_payload = {
            "user_id": "student_1",
            "session_id": "sess_cs101_01",
            "title": "计算机网络第一章（已更新）",
            "course_id": "course_cs101_adv",
            "rag_namespace": "cs_network_v2",
        }
        res_dup = await ac.post("/api/sessions", json=update_payload)
        assert res_dup.status_code == 200
        d_dup = res_dup.json()
        assert d_dup["session_id"] == "sess_cs101_01"
        assert d_dup["title"] == "计算机网络第一章（已更新）"
        assert d_dup["course_id"] == "course_cs101_adv"
        assert d_dup["rag_namespace"] == "cs_network_v2"


@pytest.mark.asyncio
async def test_get_session_detail_and_isolation(app_instance):
    """验证 GET /api/sessions/{user_id}/{session_id} 查询与用户越权防护。"""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 准备会话
        create_res = await ac.post(
            "/api/sessions",
            json={
                "user_id": "alice",
                "session_id": "sess_alice_math",
                "title": "高等数学",
                "course_id": "math_2024",
                "rag_namespace": "math_kb",
            },
        )
        assert create_res.status_code == 200

        # 正确获取
        get_res = await ac.get("/api/sessions/alice/sess_alice_math")
        assert get_res.status_code == 200
        detail = get_res.json()
        assert detail["session_id"] == "sess_alice_math"
        assert detail["course_id"] == "math_2024"
        assert detail["rag_namespace"] == "math_kb"
        assert detail["title"] == "高等数学"

        # 不存在的会话 404
        nf_res = await ac.get("/api/sessions/alice/not_existing_sess")
        assert nf_res.status_code == 404
        assert nf_res.json()["detail"] == "Session not found"

        # 跨用户访问同一 session_id 受到隔离返回 404
        cross_res = await ac.get("/api/sessions/bob/sess_alice_math")
        assert cross_res.status_code == 404
        assert cross_res.json()["detail"] == "Session not found"


@pytest.mark.asyncio
async def test_patch_session_update(app_instance):
    """验证 PATCH /api/sessions/{user_id}/{session_id} 属性更新。"""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 创建
        await ac.post(
            "/api/sessions",
            json={
                "user_id": "bob",
                "session_id": "sess_bob_physics",
                "title": "力学分析",
                "course_id": "phy_101",
                "rag_namespace": "physics_v1",
            },
        )

        # 仅修改 rag_namespace
        patch_res1 = await ac.patch(
            "/api/sessions/bob/sess_bob_physics",
            json={"rag_namespace": "physics_v2"},
        )
        assert patch_res1.status_code == 200
        p1 = patch_res1.json()
        assert p1["rag_namespace"] == "physics_v2"
        assert p1["course_id"] == "phy_101"
        assert p1["title"] == "力学分析"

        # 修改 title 与 course_id
        patch_res2 = await ac.patch(
            "/api/sessions/bob/sess_bob_physics",
            json={"title": "理论力学与电磁学", "course_id": "phy_201"},
        )
        assert patch_res2.status_code == 200
        p2 = patch_res2.json()
        assert p2["title"] == "理论力学与电磁学"
        assert p2["course_id"] == "phy_201"
        assert p2["rag_namespace"] == "physics_v2"

        # 对不存在的会话更新应返回 404
        patch_nf = await ac.patch(
            "/api/sessions/bob/no_such_session",
            json={"title": "test"},
        )
        assert patch_nf.status_code == 404


@pytest.mark.asyncio
async def test_list_and_delete_sessions(app_instance):
    """验证 GET /api/sessions/{user_id} 与 DELETE /api/sessions/{user_id}/{session_id}。"""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 创建两个会话
        await ac.post(
            "/api/sessions",
            json={
                "user_id": "charlie",
                "session_id": "s_1",
                "title": "会话一",
                "course_id": "c1",
                "rag_namespace": "ns1",
            },
        )
        await ac.post(
            "/api/sessions",
            json={
                "user_id": "charlie",
                "session_id": "s_2",
                "title": "会话二",
                "course_id": "c2",
                "rag_namespace": "ns2",
            },
        )

        # 列表查询
        list_res = await ac.get("/api/sessions/charlie")
        assert list_res.status_code == 200
        data = list_res.json()
        assert data["user_id"] == "charlie"
        sessions = data["sessions"]
        assert len(sessions) >= 2
        s1 = next(s for s in sessions if s["session_id"] == "s_1")
        assert s1["course_id"] == "c1"
        assert s1["rag_namespace"] == "ns1"
        assert s1["title"] == "会话一"

        # 删除会话
        del_res = await ac.delete("/api/sessions/charlie/s_1")
        assert del_res.status_code == 200
        assert del_res.json()["ok"] is True
        assert del_res.json()["deleted_session_id"] == "s_1"

        # 再次获取已删除的会话应为 404
        get_deleted = await ac.get("/api/sessions/charlie/s_1")
        assert get_deleted.status_code == 404

        # 列表仅剩 s_2
        list_after = await ac.get("/api/sessions/charlie")
        remaining = [s["session_id"] for s in list_after.json()["sessions"]]
        assert "s_1" not in remaining
        assert "s_2" in remaining


@pytest.mark.asyncio
async def test_concurrent_session_locking(app_instance):
    """验证多并发请求下 SessionStateManager 的锁与缓存一致性。"""
    container: AppContainer = app_instance.state.container
    state_mgr = container.state_manager

    # 并发并发获取或创建同一个 session_id
    target_sid = "concurrent_sess_test"
    tasks = [
        state_mgr.get_or_create_session(
            user_id="user_conc",
            session_id=target_sid,
            course_id=f"course_{i}",
            rag_namespace=f"ns_{i}",
        )
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    # 所有协程最终指向同一 session_id
    assert all(r.session_id == target_sid for r in results)

    # 查库验证最终状态唯一且一致
    saved = await state_mgr.get_session("user_conc", target_sid)
    assert saved is not None
    assert saved.session_id == target_sid
    assert saved.user_id == "user_conc"
    assert saved.course_id is not None
    assert saved.rag_namespace is not None
