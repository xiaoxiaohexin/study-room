"""Empirical Adversarial Stress Testing & Edge Case Verification Suite for Milestone M1.

Authored by teamwork_preview_challenger_m1_1.
Tests:
1. Concurrency, race conditions, heavy parallel session updates (50+ tasks).
2. SQL injection resistance in all fields (REST API & StateManager).
3. Unicode, emojis, special characters, whitespace, empty strings, and null handling.
4. Concurrent idempotent schema migration stress.
5. Cross-tenant isolation, cache hijacking, unauthorized cache eviction, and ID collision.
6. Extreme payload bounds (large strings).
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
from app.schemas.messages import Message
from app.storage.sqlite import SQLiteStore


@pytest.fixture
async def app_instance(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "test_adversarial.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    try:
        yield app
    finally:
        await container.aclose()


# ============================================================================
# 1. Concurrency & Race Condition Stress Tests
# ============================================================================

@pytest.mark.asyncio
async def test_concurrency_50_parallel_updates_same_session(app_instance):
    """Stress test: 50 concurrent updates on the SAME session via PATCH /api/sessions/{user}/{sid}."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "concurrent_user"
        session_id = "sess_hotspot_001"

        # 1. Create target session
        create_res = await ac.post(
            "/api/sessions",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "title": "Initial Title",
                "course_id": "course_init",
                "rag_namespace": "ns_init",
            },
        )
        assert create_res.status_code == 200

        # 2. 50 parallel updates
        async def do_patch(i: int):
            return await ac.patch(
                f"/api/sessions/{user_id}/{session_id}",
                json={
                    "title": f"Updated Title #{i}",
                    "course_id": f"course_id_{i}",
                    "rag_namespace": f"ns_{i}",
                },
            )

        responses = await asyncio.gather(*[do_patch(i) for i in range(50)])
        for idx, r in enumerate(responses):
            assert r.status_code == 200, f"Task {idx} failed with {r.status_code}: {r.text}"

        # 3. Final session state must be consistent in DB and in memory cache
        get_res = await ac.get(f"/api/sessions/{user_id}/{session_id}")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["session_id"] == session_id
        assert data["user_id"] == user_id
        assert data["title"].startswith("Updated Title #")
        assert data["course_id"].startswith("course_id_")
        assert data["rag_namespace"].startswith("ns_")


@pytest.mark.asyncio
async def test_concurrency_50_distinct_session_creations(app_instance):
    """Stress test: 50 concurrent creations of DISTINCT sessions via POST /api/sessions."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "bulk_creator"

        async def do_create(i: int):
            return await ac.post(
                "/api/sessions",
                json={
                    "user_id": user_id,
                    "session_id": f"sess_bulk_{i:03d}",
                    "title": f"Bulk Session #{i}",
                    "course_id": f"course_{i % 5}",
                    "rag_namespace": f"ns_{i % 3}",
                },
            )

        responses = await asyncio.gather(*[do_create(i) for i in range(50)])
        for idx, r in enumerate(responses):
            assert r.status_code == 200, f"Creation {idx} failed with {r.status_code}: {r.text}"

        # Verify all 50 sessions exist in listing
        list_res = await ac.get(f"/api/sessions/{user_id}")
        assert list_res.status_code == 200
        sessions = list_res.json()["sessions"]
        assert len(sessions) >= 50
        sess_ids = {s["session_id"] for s in sessions}
        for i in range(50):
            assert f"sess_bulk_{i:03d}" in sess_ids


@pytest.mark.asyncio
async def test_concurrency_interleaved_crud_and_message_appends(app_instance):
    """Stress test: Interleaved concurrent reads, updates, and message appends under StateManager."""
    container: AppContainer = app_instance.state.container
    state_mgr = container.state_manager
    user_id = "interleaved_user"
    session_id = "sess_interleaved_01"

    await state_mgr.get_or_create_session(user_id=user_id, session_id=session_id)

    async def worker_append(i: int):
        msg = Message(role="user", content=f"Hello message #{i}")
        await state_mgr.append_message(user_id=user_id, session_id=session_id, message=msg)

    async def worker_update(i: int):
        await state_mgr.update_session(
            user_id=user_id,
            session_id=session_id,
            title=f"Title v{i}",
            course_id=f"course_v{i}",
            rag_namespace=f"ns_v{i}",
        )

    async def worker_get():
        return await state_mgr.get_session(user_id=user_id, session_id=session_id)

    tasks = []
    for i in range(20):
        tasks.append(worker_append(i))
    for i in range(10):
        tasks.append(worker_update(i))
    for _ in range(20):
        tasks.append(worker_get())

    await asyncio.gather(*tasks)

    final_state = await state_mgr.get_session(user_id=user_id, session_id=session_id)
    assert final_state is not None
    assert final_state.session_id == session_id
    assert len(final_state.history) > 0
    assert len(final_state.history) <= state_mgr.history_window * 2


@pytest.mark.asyncio
async def test_concurrent_schema_migration_stress(tmp_path: Path):
    """Stress test: 10 concurrent coroutines attempting _migrate_schema() simultaneously on legacy DB."""
    legacy_db_path = tmp_path / "concurrent_mig.db"

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
        await db.commit()

    store = SQLiteStore(legacy_db_path)
    store._db = await aiosqlite.connect(legacy_db_path)
    store._db.row_factory = aiosqlite.Row

    try:
        results = await asyncio.gather(
            *[store._migrate_schema() for _ in range(10)],
            return_exceptions=True,
        )
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent migrations produced exceptions: {exceptions}"
    finally:
        await store.aclose()


# ============================================================================
# 2. SQL Injection Resistance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_sql_injection_resistance_via_rest_api(app_instance):
    """Verify that malicious SQL injection strings in all fields are handled safely via REST API."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        injection_payloads = [
            "'; DROP TABLE sessions; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT session_id, user_id, title, course_id, rag_namespace, created_at, updated_at FROM sessions --",
            "'; DELETE FROM messages; --",
            "\" OR \"\"=\"",
            "` OR 1=1 --",
        ]

        for idx, payload in enumerate(injection_payloads):
            sid = f"inject_sess_{idx}"
            res = await ac.post(
                "/api/sessions",
                json={
                    "user_id": f"inject_user_{idx}",
                    "session_id": sid,
                    "title": payload,
                    "course_id": payload,
                    "rag_namespace": payload,
                },
            )
            assert res.status_code == 200, f"Injection payload failed creation: {payload}"
            data = res.json()
            assert data["title"] == payload
            assert data["course_id"] == payload
            assert data["rag_namespace"] == payload

            patch_res = await ac.patch(
                f"/api/sessions/inject_user_{idx}/{sid}",
                json={"title": f"Updated {payload}", "course_id": payload},
            )
            assert patch_res.status_code == 200
            patch_data = patch_res.json()
            assert patch_data["title"] == f"Updated {payload}"

        container: AppContainer = app_instance.state.container
        cur = await container.sqlite.conn.execute("SELECT count(*) as cnt FROM sessions")
        row = await cur.fetchone()
        assert row is not None
        assert row["cnt"] >= len(injection_payloads)


@pytest.mark.asyncio
async def test_sql_injection_resistance_via_state_manager(app_instance):
    """Verify SessionStateManager parameterized queries withstand SQL injection."""
    container: AppContainer = app_instance.state.container
    state_mgr = container.state_manager

    malicious_user = "user' OR 1=1; --"
    malicious_sid = "sess' UNION SELECT * FROM sessions; --"
    malicious_course = "course' OR 'a'='a"
    malicious_ns = "ns'; DROP TABLE messages; --"

    state = await state_mgr.get_or_create_session(
        user_id=malicious_user,
        session_id=malicious_sid,
        course_id=malicious_course,
        rag_namespace=malicious_ns,
        title="SQLi Test",
    )
    assert state.user_id == malicious_user
    assert state.session_id == malicious_sid
    assert state.course_id == malicious_course
    assert state.rag_namespace == malicious_ns

    fetched = await state_mgr.get_session(user_id=malicious_user, session_id=malicious_sid)
    assert fetched is not None
    assert fetched.session_id == malicious_sid
    assert fetched.rag_namespace == malicious_ns

    cross_fetched = await state_mgr.get_session(user_id="other_user", session_id=malicious_sid)
    assert cross_fetched is None

    sessions = await state_mgr.list_sessions(user_id=malicious_user)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == malicious_sid


# ============================================================================
# 3. Unicode, Special Characters, Emojis & Null/Empty Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_unicode_and_emoji_handling(app_instance):
    """Verify full fidelity for CJK characters, emojis, math symbols, and mixed strings."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "用户_123_👩‍💻"
        session_id = "会话_🌟_2026_∫f(x)dx"
        title = "高数与量子力学 🚀 [第4章] · 极限与连续性 📚"
        course_id = "课程_高等数学_2026_αβγ"
        rag_namespace = "知识库_高数_v1.0_🎯"

        res = await ac.post(
            "/api/sessions",
            json={
                "user_id": user_id,
                "session_id": session_id,
                "title": title,
                "course_id": course_id,
                "rag_namespace": rag_namespace,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == user_id
        assert data["session_id"] == session_id
        assert data["title"] == title
        assert data["course_id"] == course_id
        assert data["rag_namespace"] == rag_namespace

        get_res = await ac.get(f"/api/sessions/{user_id}/{session_id}")
        assert get_res.status_code == 200
        get_data = get_res.json()
        assert get_data["title"] == title
        assert get_data["course_id"] == course_id
        assert get_data["rag_namespace"] == rag_namespace


@pytest.mark.asyncio
async def test_null_and_empty_strings_handling(app_instance):
    """Verify behavior with None, empty strings, and default fallback mechanisms."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "test_null_user"

        res1 = await ac.post("/api/sessions", json={"user_id": user_id, "session_id": "s_null_1"})
        assert res1.status_code == 200
        d1 = res1.json()
        assert d1["course_id"] is None
        assert d1["rag_namespace"] == "default"

        res2 = await ac.post(
            "/api/sessions",
            json={
                "user_id": user_id,
                "session_id": "s_null_2",
                "course_id": "",
                "rag_namespace": "",
            },
        )
        assert res2.status_code == 200
        d2 = res2.json()
        assert d2["course_id"] == ""
        # rag_namespace empty string should fallback to default in POST
        assert d2["rag_namespace"] == "default"

        res3 = await ac.patch(
            f"/api/sessions/{user_id}/s_null_2",
            json={"title": "", "course_id": None},
        )
        assert res3.status_code == 200
        d3 = res3.json()
        assert d3["title"] == ""


@pytest.mark.asyncio
async def test_large_payload_boundaries(app_instance):
    """Verify handling of extreme string payloads (50,000 characters)."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        large_title = "A" * 50000
        large_course = "C" * 10000
        large_ns = "N" * 10000

        res = await ac.post(
            "/api/sessions",
            json={
                "user_id": "large_payload_user",
                "session_id": "sess_large_payload",
                "title": large_title,
                "course_id": large_course,
                "rag_namespace": large_ns,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["title"] == large_title
        assert data["course_id"] == large_course
        assert data["rag_namespace"] == large_ns


# ============================================================================
# 4. Cross-Tenant Isolation & Security Boundary Tests
# ============================================================================

@pytest.mark.asyncio
async def test_cross_tenant_session_hijack_attempt(app_instance):
    """Adversarial check: User B tries to claim or overwrite User A's session_id."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # User A creates a session
        res_a = await ac.post(
            "/api/sessions",
            json={
                "user_id": "alice",
                "session_id": "alice_secret_sess",
                "title": "Alice's Secret Exam Notes",
                "course_id": "cs_exam",
                "rag_namespace": "alice_kb",
            },
        )
        assert res_a.status_code == 200

        # User B attempts to delete Alice's session via DELETE /api/sessions/bob/alice_secret_sess
        del_b = await ac.delete("/api/sessions/bob/alice_secret_sess")
        assert del_b.status_code == 200

        # Verify Alice's session is STILL intact in DB and accessible by Alice
        get_a = await ac.get("/api/sessions/alice/alice_secret_sess")
        assert get_a.status_code == 200
        assert get_a.json()["title"] == "Alice's Secret Exam Notes"

        # User B attempts to modify Alice's session via PATCH
        patch_b = await ac.patch(
            "/api/sessions/bob/alice_secret_sess",
            json={"title": "Hacked Title by Bob"},
        )
        assert patch_b.status_code == 404

        # Alice's session must retain original title
        get_a2 = await ac.get("/api/sessions/alice/alice_secret_sess")
        assert get_a2.json()["title"] == "Alice's Secret Exam Notes"

        # User B attempts to POST /api/sessions with Alice's session_id
        # In a robust API, attempting to claim an existing session belonging to another user
        # must return a client error (400/403/409), NOT an unhandled 500 crash.
        post_b = await ac.post(
            "/api/sessions",
            json={
                "user_id": "bob",
                "session_id": "alice_secret_sess",
                "title": "Bob's Hijacked Session",
                "course_id": "bob_course",
                "rag_namespace": "bob_ns",
            },
        )
        assert post_b.status_code in (400, 403, 409), (
            f"Vulnerability: POST /api/sessions returned status {post_b.status_code} "
            f"instead of client error (400/403/409) when attempting to create existing session_id of another user. "
            f"Response body: {post_b.text}"
        )


@pytest.mark.asyncio
async def test_cross_tenant_cache_eviction_vulnerability(app_instance):
    """Adversarial check: User B calling DELETE on Alice's session should NOT evict Alice's session from StateManager cache."""
    container: AppContainer = app_instance.state.container
    state_mgr = container.state_manager

    # Alice creates and caches a session
    await state_mgr.get_or_create_session(
        user_id="alice",
        session_id="alice_cached_sess",
        title="Alice Active Session",
    )
    assert "alice_cached_sess" in state_mgr._cache

    # Bob attempts to delete Alice's session
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        del_res = await ac.delete("/api/sessions/bob/alice_cached_sess")
        assert del_res.status_code == 200

    # Vulnerability check: Bob's unauthorized delete should NOT have purged Alice's in-memory session cache!
    assert "alice_cached_sess" in state_mgr._cache, (
        "Vulnerability found: User B successfully evicted User A's session from StateManager._cache!"
    )


@pytest.mark.asyncio
async def test_patch_empty_rag_namespace_inconsistency(app_instance):
    """Verify whether PATCH setting rag_namespace to empty string '' causes cache vs DB inconsistency."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        user_id = "user_ns_test"
        session_id = "sess_ns_empty"

        await ac.post(
            "/api/sessions",
            json={"user_id": user_id, "session_id": session_id, "rag_namespace": "math_v1"},
        )

        patch_res = await ac.patch(
            f"/api/sessions/{user_id}/{session_id}",
            json={"rag_namespace": ""},
        )
        assert patch_res.status_code == 200
        patch_data = patch_res.json()

        # Evict cache to force DB reload
        container: AppContainer = app_instance.state.container
        container.state_manager._cache.pop(session_id, None)

        get_res = await ac.get(f"/api/sessions/{user_id}/{session_id}")
        assert get_res.status_code == 200
        get_data = get_res.json()

        # Both must agree
        assert patch_data["rag_namespace"] == get_data["rag_namespace"], (
            f"Inconsistency: PATCH returned rag_namespace='{patch_data['rag_namespace']}', "
            f"but subsequent DB reload returned rag_namespace='{get_data['rag_namespace']}'"
        )
