"""Phase P0-A tests: learning-data ownership and course/namespace isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.learning.store import LearningStore
from app.main import create_app
from app.storage.sqlite import SQLiteStore


@pytest.fixture
async def store(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "learning_iso.db")
    await sqlite.initialize()
    yield LearningStore(sqlite), sqlite
    await sqlite.aclose()


@pytest.fixture
async def api_client(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "learning_iso_api.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, container
    await container.aclose()


class TestStoreOwnership:
    @pytest.mark.asyncio
    async def test_ensure_course_creates_row_and_rejects_foreign_owner(self, store):
        learning, _ = store
        course = await learning.ensure_course("alice", "math_calc")
        assert course["id"] == "math_calc"
        assert course["user_id"] == "alice"

        # Idempotent for the owner.
        again = await learning.ensure_course("alice", "math_calc")
        assert again["user_id"] == "alice"

        with pytest.raises(PermissionError):
            await learning.ensure_course("bob", "math_calc")

        # `default` is virtual: no course row is required and every user shares the id.
        virtual = await learning.ensure_course("bob", "default")
        assert virtual["virtual"] is True

    @pytest.mark.asyncio
    async def test_name_lookup_is_scoped_to_course_and_user(self, store):
        learning, _ = store
        kp_a = await learning.upsert_knowledge_point(
            course_id="bio_alice", name="光合作用", user_id="alice"
        )
        kp_b = await learning.upsert_knowledge_point(
            course_id="bio_bob", name="光合作用", user_id="bob"
        )
        assert kp_a["id"] != kp_b["id"]

        found_bob = await learning.get_knowledge_point(
            "光合作用", course_id="bio_bob", user_id="bob"
        )
        assert found_bob is not None
        assert found_bob["id"] == kp_b["id"]

        # Another course or another user must not leak the row.
        assert (
            await learning.get_knowledge_point("光合作用", course_id="chem", user_id="bob")
            is None
        )
        assert (
            await learning.get_knowledge_point(
                "光合作用", course_id="bio_alice", user_id="carol"
            )
            is None
        )
        # Bob cannot look up Alice's knowledge point by name in her course either.
        assert (
            await learning.get_knowledge_point(
                "光合作用", course_id="bio_alice", user_id="bob"
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_cross_course_update_by_kp_id_is_rejected(self, store):
        learning, _ = store
        kp = await learning.upsert_knowledge_point(
            course_id="course_a", name="牛顿第一定律", user_id="alice"
        )
        with pytest.raises(ValueError):
            await learning.upsert_knowledge_point(
                course_id="course_b",
                name="被跨课程改写",
                kp_id=kp["id"],
                user_id="alice",
            )
        row = await learning.get_knowledge_point(
            kp["id"], course_id="course_a", user_id="alice"
        )
        assert row is not None
        assert row["name"] == "牛顿第一定律"
        assert row["course_id"] == "course_a"

    @pytest.mark.asyncio
    async def test_cross_course_and_cross_user_relations_are_rejected(self, store):
        learning, _ = store
        kp_a = await learning.upsert_knowledge_point(
            course_id="course_a", name="概念 A", user_id="alice"
        )
        kp_b = await learning.upsert_knowledge_point(
            course_id="course_b", name="概念 B", user_id="alice"
        )
        kp_other_user = await learning.upsert_knowledge_point(
            course_id="course_a_bob", name="他人概念", user_id="bob"
        )

        with pytest.raises(ValueError):
            await learning.upsert_relation(
                src_id=kp_a["id"], rel_type="causal", dst_id=kp_b["id"], user_id="alice"
            )
        with pytest.raises(PermissionError):
            await learning.upsert_relation(
                src_id=kp_a["id"],
                rel_type="causal",
                dst_id=kp_other_user["id"],
                user_id="alice",
            )

        # Dangling endpoints still allowed for offline canvas sync.
        rel = await learning.upsert_relation(
            src_id="kp_future_1",
            rel_type="similar",
            dst_id="kp_future_2",
            user_id="alice",
        )
        assert rel["src_id"] == "kp_future_1"

    @pytest.mark.asyncio
    async def test_lists_revisions_and_rollback_are_user_scoped(self, store):
        learning, _ = store
        kp = await learning.upsert_knowledge_point(
            course_id="course_x_alice", name="泰勒展开", content="v1", user_id="alice"
        )
        await learning.upsert_knowledge_point(
            course_id="course_x_alice",
            name="泰勒展开",
            content="v2",
            kp_id=kp["id"],
            user_id="alice",
        )
        await learning.upsert_knowledge_point(
            course_id="course_x_bob", name="鲍勃知识点", content="bob", user_id="bob"
        )

        alice_kps = await learning.list_knowledge_points(
            course_id="course_x_alice", user_id="alice"
        )
        bob_kps = await learning.list_knowledge_points(
            course_id="course_x_bob", user_id="bob"
        )
        assert [k["name"] for k in alice_kps] == ["泰勒展开"]
        assert [k["name"] for k in bob_kps] == ["鲍勃知识点"]

        assert len(await learning.get_revisions(kp["id"], user_id="alice")) == 2
        assert await learning.get_revisions(kp["id"], user_id="bob") == []

        with pytest.raises(KeyError):
            await learning.rollback_knowledge_point(kp["id"], 1, user_id="bob")

        rolled = await learning.rollback_knowledge_point(kp["id"], 1, user_id="alice")
        assert rolled["content"] == "v1"

    @pytest.mark.asyncio
    async def test_delete_knowledge_point_cascades_relations_and_mastery(self, store):
        learning, sqlite = store
        kp_a = await learning.upsert_knowledge_point(
            course_id="course_del", name="A", user_id="alice"
        )
        kp_b = await learning.upsert_knowledge_point(
            course_id="course_del", name="B", prereq_ids=[kp_a["id"]], user_id="alice"
        )
        await learning.upsert_relation(
            src_id=kp_a["id"], rel_type="causal", dst_id=kp_b["id"], user_id="alice"
        )

        assert await learning.delete_knowledge_point(kp_b["id"], user_id="alice") is True

        cur = await sqlite.conn.execute(
            "SELECT count(*) AS cnt FROM kp_relations WHERE src_id = ? OR dst_id = ?",
            (kp_b["id"], kp_b["id"]),
        )
        assert (await cur.fetchone())["cnt"] == 0
        cur = await sqlite.conn.execute(
            "SELECT count(*) AS cnt FROM mastery WHERE kp_id = ?", (kp_b["id"],)
        )
        assert (await cur.fetchone())["cnt"] == 0
        cur = await sqlite.conn.execute(
            "SELECT count(*) AS cnt FROM kp_prereq WHERE kp_id = ? OR prereq_id = ?",
            (kp_b["id"], kp_b["id"]),
        )
        assert (await cur.fetchone())["cnt"] == 0

        # Deleting someone else's knowledge point must not work.
        assert await learning.delete_knowledge_point(kp_a["id"], user_id="bob") is False


class TestApiOwnership:
    @pytest.mark.asyncio
    async def test_knowledge_api_isolated_by_user_header(self, api_client):
        client, _ = api_client
        resp = await client.post(
            "/api/courses/default/knowledge",
            headers={"X-User-Id": "alice"},
            json={"name": "Alice 的知识点"},
        )
        assert resp.status_code == 200
        alice_id = resp.json()["id"]

        # Other user cannot see or modify it.
        listed = await client.get(
            "/api/courses/default/knowledge", headers={"X-User-Id": "bob"}
        )
        assert listed.json()["count"] == 0

        hijack = await client.post(
            "/api/courses/default/knowledge",
            headers={"X-User-Id": "bob"},
            json={"id": alice_id, "name": "劫持"},
        )
        assert hijack.status_code == 403

        # Owner can still update it.
        update = await client.post(
            "/api/courses/default/knowledge",
            headers={"X-User-Id": "alice"},
            json={"id": alice_id, "name": "Alice 的知识点", "content": "更新"},
        )
        assert update.status_code == 200
        assert update.json()["version"] == 2

    @pytest.mark.asyncio
    async def test_cross_course_relation_api_returns_conflict(self, api_client):
        client, _ = api_client
        kp_a = await client.post(
            "/api/courses/course_a/knowledge",
            headers={"X-User-Id": "alice"},
            json={"name": "A"},
        )
        kp_b = await client.post(
            "/api/courses/course_b/knowledge",
            headers={"X-User-Id": "alice"},
            json={"name": "B"},
        )
        resp = await client.post(
            "/api/courses/course_a/knowledge/relations",
            headers={"X-User-Id": "alice"},
            json={
                "src_id": kp_a.json()["id"],
                "rel_type": "causal",
                "dst_id": kp_b.json()["id"],
            },
        )
        assert resp.status_code == 409
