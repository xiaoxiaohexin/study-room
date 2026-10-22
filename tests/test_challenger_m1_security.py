"""Empirical Adversarial Security & Isolation Test Suite for Milestone M1.

Authored by Empirical Challenger (teamwork_preview_challenger_m1_2).
Verifies:
1. Multi-tenant isolation: Cross-user session access, modification, and hijacking.
2. In-memory cache integrity: Cache poisoning / desync on database failure.
3. Cross-user cache eviction: Bob evicting Alice's cached session via DELETE.
4. Deletion cascades & orphan messages: Unfiltered message loading & lack of lock in append_message.
5. Input boundary handling: SQL injection strings, unicode namespaces, empty IDs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.main import create_app
from app.schemas.messages import Message


@pytest.fixture
async def app_instance(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "test_sec.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app
    await container.aclose()


# ============================================================================
# 1. Multi-Tenant Isolation: Read, Update, and Hijack Attempts
# ============================================================================

@pytest.mark.asyncio
async def test_cross_tenant_read_and_patch_rejection(app_instance):
    """验证租户绝对隔离：用户 B 无法读取或 PATCH 修改用户 A 的会话。"""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Alice 创建专属私有会话
        create_res = await ac.post(
            "/api/sessions",
            json={
                "user_id": "alice",
                "session_id": "alice_secret_sess",
                "title": "Alice 的机密会话",
                "course_id": "course_crypto",
                "rag_namespace": "crypto_ns",
            },
        )
        assert create_res.status_code == 200

        # Bob 尝试通过 GET 读取 Alice 的会话 -> 必须 404
        bob_read = await ac.get("/api/sessions/bob/alice_secret_sess")
        assert bob_read.status_code == 404, f"Cross-tenant read leaked! Status: {bob_read.status_code}"

        # Bob 尝试通过 PATCH 篡改 Alice 的会话 -> 必须 404
        bob_patch = await ac.patch(
            "/api/sessions/bob/alice_secret_sess",
            json={"title": "Hacked Title", "rag_namespace": "hacked_ns"},
        )
        assert bob_patch.status_code == 404, f"Cross-tenant patch succeeded! Status: {bob_patch.status_code}"

        # 验证 Alice 的会话未被篡改
        alice_check = await ac.get("/api/sessions/alice/alice_secret_sess")
        assert alice_check.status_code == 200
        data = alice_check.json()
        assert data["title"] == "Alice 的机密会话"
        assert data["rag_namespace"] == "crypto_ns"


@pytest.mark.asyncio
async def test_cross_tenant_hijack_via_post_fails_or_conflicts(app_instance):
    """验证用户 B 试图使用用户 A 已占用的 session_id 创建会话时的行为（防会话劫持）。"""
    transport = ASGITransport(app=app_instance, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Alice 创建会话
        res_a = await ac.post(
            "/api/sessions",
            json={
                "user_id": "alice",
                "session_id": "sess_shared_target",
                "title": "Alice original",
                "course_id": "course_a",
                "rag_namespace": "ns_a",
            },
        )
        assert res_a.status_code == 200

        # Bob 尝试使用相同的 session_id 调用 POST /api/sessions 强行覆盖/接入
        # 由于 session_id 为 Primary Key，且 user_id 不匹配，后端应拒绝并不应覆盖 Alice 的数据
        res_b = await ac.post(
            "/api/sessions",
            json={
                "user_id": "bob",
                "session_id": "sess_shared_target",
                "title": "Bob hijacked",
                "course_id": "course_b",
                "rag_namespace": "ns_b",
            },
        )
        # 观察 HTTP 状态码：
        # 理想设计应返回 409 Conflict 或 403 Forbidden；当前实现抛出未捕获的 sqlite3.IntegrityError 返回 500
        assert res_b.status_code != 200, "CRITICAL VULNERABILITY: Bob successfully hijacked Alice's session_id with status 200!"
        # 记录实际状态码
        assert res_b.status_code in (400, 403, 409), (
            f"API CRASH ON CROSS-TENANT CONFLICT: Expected HTTP 409/403/400 but got HTTP {res_b.status_code}. "
            "Backend crashed with unhandled sqlite3.IntegrityError instead of validating existing session ownership!"
        )

        # 进一步验证 Alice 的原始会话数据完整性
        alice_verify = await ac.get("/api/sessions/alice/sess_shared_target")
        assert alice_verify.status_code == 200
        alice_data = alice_verify.json()
        assert alice_data["user_id"] == "alice"
        assert alice_data["title"] == "Alice original"
        assert alice_data["rag_namespace"] == "ns_a"


# ============================================================================
# 2. In-Memory Cache Integrity & Rollback Under Database Failure
# ============================================================================

@pytest.mark.asyncio
async def test_cache_poisoning_vulnerability_on_db_update_failure(app_instance):
    """测试脆弱点验证：当会话处于内存缓存中时，若后续数据库 UPDATE 执行失败，内存缓存是否会被提前污染。"""
    container: AppContainer = app_instance.state.container
    mgr = container.state_manager

    # 1. 建立初始会话并缓存
    s = await mgr.get_or_create_session(
        user_id="user_test",
        session_id="cache_sync_sess_1",
        title="Valid Title",
        course_id="c1",
        rag_namespace="ns_valid",
    )
    assert s.session_id in mgr._cache
    assert mgr._cache[s.session_id].title == "Valid Title"

    # 2. 模拟数据库故障（例如 DB 被锁定或断开）
    orig_execute = mgr.sqlite.conn.execute

    async def faulty_execute(sql, *args, **kwargs):
        if "UPDATE sessions SET" in sql:
            raise aiosqlite.OperationalError("database is locked (simulated failure)")
        return await orig_execute(sql, *args, **kwargs)

    # 3. 在故障情况下调用 get_or_create_session 尝试更新会话属性
    with patch.object(mgr.sqlite.conn, "execute", side_effect=faulty_execute):
        with pytest.raises(aiosqlite.OperationalError):
            await mgr.get_or_create_session(
                user_id="user_test",
                session_id="cache_sync_sess_1",
                title="Poisoned Title",
                course_id="c_poisoned",
                rag_namespace="ns_poisoned",
            )

    # 4. 检查内存缓存与数据库的一致性
    # 数据库中的实际状态（通过独立连接或原始 execute 读取）
    cur = await orig_execute("SELECT title, rag_namespace FROM sessions WHERE session_id = ?", ("cache_sync_sess_1",))
    db_row = await cur.fetchone()
    assert db_row["title"] == "Valid Title", "DB was unexpectedly modified!"
    assert db_row["rag_namespace"] == "ns_valid"

    # 检查内存缓存 _cache:
    # 评估：如果 state 对象在 DB 执行前被就地修改，则此处 cache 会被污染
    cached_title = mgr._cache["cache_sync_sess_1"].title
    cached_ns = mgr._cache["cache_sync_sess_1"].rag_namespace

    # 断言缓存一致性：DB 操作失败后，缓存必须保留原始状态，不能残留未持久化的修改！
    cache_poisoned = (cached_title != "Valid Title") or (cached_ns != "ns_valid")
    assert not cache_poisoned, (
        f"CACHE POISONING CONFIRMED: In-memory cache contains uncommitted values "
        f"(title='{cached_title}', rag_namespace='{cached_ns}') while DB has "
        f"(title='{db_row['title']}', rag_namespace='{db_row['rag_namespace']}')."
    )


# ============================================================================
# 3. Cross-Tenant Cache Eviction via Unauthorized DELETE
# ============================================================================

@pytest.mark.asyncio
async def test_cross_tenant_cache_eviction_via_delete(app_instance):
    """测试隔离缺陷：用户 B 调用 DELETE 删除不存在于自己名下的 Alice 会话，是否会越权将 Alice 的内存缓存驱逐。"""
    transport = ASGITransport(app=app_instance)
    container: AppContainer = app_instance.state.container
    mgr = container.state_manager

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Alice 创建会话，此时放入 mgr._cache
        await ac.post(
            "/api/sessions",
            json={
                "user_id": "alice",
                "session_id": "alice_cached_sess",
                "title": "Alice Cache",
            },
        )
        assert "alice_cached_sess" in mgr._cache
        assert mgr._cache["alice_cached_sess"].user_id == "alice"

        # Bob 尝试调用 DELETE 接口删除 Alice 的 session_id
        del_res = await ac.delete("/api/sessions/bob/alice_cached_sess")
        assert del_res.status_code == 200

        # 检查 Alice 的 DB 数据：确实没有被删除
        cur = await mgr.sqlite.conn.execute("SELECT * FROM sessions WHERE session_id = ?", ("alice_cached_sess",))
        row = await cur.fetchone()
        assert row is not None, "Alice's DB row was deleted by Bob!"

        # 检查 Alice 的缓存状态：
        # 如果 Bob 的 delete 操作触发了 `_cache.pop("alice_cached_sess", None)`，
        # 则 Alice 的热点内存缓存被恶意第三方清空！
        assert "alice_cached_sess" in mgr._cache, (
            "CROSS-TENANT CACHE EVICTION CONFIRMED: Bob's unauthorized DELETE evicted "
            "Alice's active session from in-memory cache!"
        )


# ============================================================================
# 4. Message Query Isolation & Orphan Message Safety
# ============================================================================

@pytest.mark.asyncio
async def test_messages_query_user_id_filtering_and_cascade(app_instance):
    """测试消息历史的租户过滤与级联删除：
    若 messages 表中存在相同 session_id 但不同 user_id 的消息，加载会话时是否会发生越权混入，
    以及删除会话时是否会残留孤儿消息。
    """
    container: AppContainer = app_instance.state.container
    mgr = container.state_manager

    # 1. Alice 创建会话
    await mgr.get_or_create_session(user_id="alice", session_id="sess_msg_iso")
    await mgr.append_message("alice", "sess_msg_iso", Message(role="user", content="Alice query"))

    # 2. 模拟向 messages 写入了属于 bob 的消息（跨用户消息注入）
    await mgr.sqlite.conn.execute(
        "INSERT INTO messages (id, session_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("msg_bob_injected", "sess_msg_iso", "bob", "user", "Bob unauthorized text", 1000.0),
    )
    await mgr.sqlite.conn.commit()

    # 清空缓存强制从 DB 重新加载
    mgr._cache.clear()

    # 3. Alice 重新读取自己的会话
    reloaded = await mgr.get_session("alice", "sess_msg_iso")
    assert reloaded is not None

    # 检查 Alice 的 history 中是否混入了 Bob 的消息
    bob_in_history = any(m.content == "Bob unauthorized text" for m in reloaded.history)
    assert not bob_in_history, (
        "CROSS-TENANT MESSAGE LEAKAGE CONFIRMED: Bob's message was included in Alice's "
        "history because messages query lacks 'AND user_id = ?'!"
    )

    # 4. 级联删除测试：Alice 删除会话
    await mgr.delete_session("alice", "sess_msg_iso")

    # 检查 messages 表中是否残留孤儿数据
    cur = await mgr.sqlite.conn.execute("SELECT * FROM messages WHERE session_id = ?", ("sess_msg_iso",))
    orphan_msgs = await cur.fetchall()
    assert len(orphan_msgs) == 0, (
        f"ORPHAN MESSAGES CONFIRMED: {len(orphan_msgs)} messages still exist in messages table "
        f"after session was deleted!"
    )


# ============================================================================
# 5. append_message Lack of Concurrency Lock & Orphan Creation
# ============================================================================

@pytest.mark.asyncio
async def test_append_message_on_deleted_session_creates_orphans(app_instance):
    """测试 append_message 未校验 session 存在性导致孤儿记录产生。"""
    container: AppContainer = app_instance.state.container
    mgr = container.state_manager

    # 对一个完全不存在 / 已删除的 session_id 执行 append_message
    non_existent_sid = "ghost_session_999"
    await mgr.append_message("alice", non_existent_sid, Message(role="user", content="orphan message"))

    # 查询 messages 表
    cur = await mgr.sqlite.conn.execute("SELECT * FROM messages WHERE session_id = ?", (non_existent_sid,))
    orphans = await cur.fetchall()

    # 查询 sessions 表
    cur_sess = await mgr.sqlite.conn.execute("SELECT * FROM sessions WHERE session_id = ?", (non_existent_sid,))
    sess_row = await cur_sess.fetchone()

    # 评估：sessions 表无此记录，但 messages 表却持久化了该消息
    assert sess_row is None
    # 若允许无 session 孤儿写入，记录该现象
    assert len(orphans) == 0, (
        f"ORPHAN INSERT CONFIRMED: append_message inserted {len(orphans)} message(s) "
        f"for a non-existent session_id without foreign key or existence validation!"
    )


@pytest.mark.asyncio
async def test_concurrent_delete_and_append_race_condition(app_instance):
    """测试并发 delete_session 与 append_message 竞态条件：由于 append_message 未加锁，导致孤儿消息残留。"""
    container: AppContainer = app_instance.state.container
    mgr = container.state_manager

    await mgr.get_or_create_session(user_id="alice", session_id="race_sess_01")

    # 并发执行 delete_session 与多个 append_message
    async def do_delete():
        await asyncio.sleep(0.002)
        await mgr.delete_session("alice", "race_sess_01")

    async def do_append(idx):
        await asyncio.sleep(0.001 * idx)
        await mgr.append_message("alice", "race_sess_01", Message(role="user", content=f"concurrent msg {idx}"))

    await asyncio.gather(
        do_delete(),
        do_append(1),
        do_append(2),
        do_append(3),
    )

    # 检查数据库中 session 是否被删除
    cur_sess = await mgr.sqlite.conn.execute("SELECT * FROM sessions WHERE session_id = 'race_sess_01'")
    sess_row = await cur_sess.fetchone()
    cur_msg = await mgr.sqlite.conn.execute("SELECT * FROM messages WHERE session_id = 'race_sess_01'")
    msg_rows = await cur_msg.fetchall()

    if sess_row is None:
        # 如果 session 已被删除，messages 中绝不能有任何残留孤儿消息！
        assert len(msg_rows) == 0, (
            f"CONCURRENCY RACE CORRUPTION: Session was deleted, but {len(msg_rows)} orphan messages "
            f"were written concurrently because append_message does not acquire session lock!"
        )


# ============================================================================
# 6. Boundary Conditions: Special Characters & Injection
# ============================================================================

@pytest.mark.asyncio
async def test_special_characters_and_sql_injection_safety(app_instance):
    """测试特殊字符、长字符串及 SQL 注入模式下的鲁棒性。"""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "user_id": "user' OR '1'='1",
            "session_id": "sess_inj'; DROP TABLE sessions; --",
            "title": "'); DELETE FROM messages; --",
            "course_id": "course' UNION SELECT * FROM users --",
            "rag_namespace": "ns_🚀_unicode_中文_!@#$%%^&*()_+",
        }
        res = await ac.post("/api/sessions", json=payload)
        assert res.status_code == 200
        d = res.json()
        assert d["user_id"] == "user' OR '1'='1"
        assert d["rag_namespace"] == "ns_🚀_unicode_中文_!@#$%%^&*()_+"

        # 验证 sessions 表仍然完整存在
        container: AppContainer = app_instance.state.container
        cur = await container.sqlite.conn.execute("SELECT COUNT(*) as c FROM sessions")
        count = (await cur.fetchone())["c"]
        assert count >= 1
