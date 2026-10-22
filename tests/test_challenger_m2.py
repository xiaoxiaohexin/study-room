"""Adversarial Verification & Stress Test Suite for Milestone 2.

Authored by Empirical Challenger M2-1.
Adversarially verifies and stress-tests:
1. Duplicate relation insertion idempotency (INSERT OR REPLACE behavior)
2. Self-loop relations (src_id == dst_id)
3. Non-existent knowledge point IDs in relations (dangling/ghost references)
4. Circular dependency graphs (A -> B -> C -> A) and query termination safety
5. Batch inserts of 50+ semantic edges (scale & concurrency stress)
6. Multi-step version history & non-destructive rollback stress
7. Mastery auto-enrollment SM-2 persistence & progress protection
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.container import AppContainer
from app.learning.store import LearningStore
from app.main import create_app
from app.storage.sqlite import SQLiteStore


@pytest.fixture
async def challenger_sqlite(tmp_path: Path):
    db_path = tmp_path / "challenger_m2.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    yield store
    await store.aclose()


@pytest.fixture
async def challenger_app(tmp_path: Path):
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "challenger_api.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield app
    await container.aclose()


# ============================================================================
# 1. Duplicate Relation Insertion Idempotency
# ============================================================================

class TestDuplicateRelationInsertionIdempotency:
    """Stress-test idempotent upsert behavior on duplicate semantic edges."""

    @pytest.mark.asyncio
    async def test_repeated_relation_upsert_updates_metadata_without_duplicate_rows(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        kp_a = await store.upsert_knowledge_point("course_math", "导数概念")
        kp_b = await store.upsert_knowledge_point("course_math", "导数几何意义")

        # First insert
        rel1 = await store.upsert_relation(
            src_id=kp_a["id"],
            rel_type="prerequisite",
            dst_id=kp_b["id"],
            metadata={"weight": 1.0, "note": "initial"},
        )
        assert rel1["src_id"] == kp_a["id"]
        assert rel1["metadata"]["weight"] == 1.0

        # Second insert: identical (src_id, rel_type, dst_id) with updated metadata
        rel2 = await store.upsert_relation(
            src_id=kp_a["id"],
            rel_type="prerequisite",
            dst_id=kp_b["id"],
            metadata={"weight": 2.5, "note": "updated_metadata"},
        )
        assert rel2["metadata"]["weight"] == 2.5

        # Inspect database table directly: exactly 1 row should exist
        cur = await challenger_sqlite.conn.execute(
            "SELECT count(*) as cnt FROM kp_relations WHERE src_id = ? AND rel_type = ? AND dst_id = ?",
            (kp_a["id"], "prerequisite", kp_b["id"]),
        )
        row = await cur.fetchone()
        assert row["cnt"] == 1

        # Direct table inspection of metadata_json
        cur2 = await challenger_sqlite.conn.execute(
            "SELECT metadata_json FROM kp_relations WHERE src_id = ? AND rel_type = ? AND dst_id = ?",
            (kp_a["id"], "prerequisite", kp_b["id"]),
        )
        meta_row = await cur2.fetchone()
        saved_meta = json.loads(meta_row["metadata_json"])
        assert saved_meta["weight"] == 2.5
        assert saved_meta["note"] == "updated_metadata"

        # Verify kp_prereq dual-write table has exactly 1 row
        cur_p = await challenger_sqlite.conn.execute(
            "SELECT count(*) as cnt FROM kp_prereq WHERE kp_id = ? AND prereq_id = ?",
            (kp_b["id"], kp_a["id"]),
        )
        p_row = await cur_p.fetchone()
        assert p_row["cnt"] == 1

    @pytest.mark.asyncio
    async def test_duplicate_relations_in_batch_upsert_payload(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        kp_root = await store.upsert_knowledge_point("cs101", "基础编程")
        kp_target = await store.upsert_knowledge_point("cs101", "高级指针")

        # Pass multiple duplicated relation entries in a single upsert_knowledge_point call
        duplicate_relations = [
            {"src_id": kp_root["id"], "dst_id": kp_target["id"], "rel_type": "causal", "metadata": {"v": 1}},
            {"src_id": kp_root["id"], "dst_id": kp_target["id"], "rel_type": "causal", "metadata": {"v": 2}},
            {"src_id": kp_root["id"], "dst_id": kp_target["id"], "rel_type": "causal", "metadata": {"v": 3}},
        ]

        await store.upsert_knowledge_point(
            course_id="cs101",
            name="高级指针",
            kp_id=kp_target["id"],
            relations=duplicate_relations,
        )

        cur = await challenger_sqlite.conn.execute(
            "SELECT count(*) as cnt FROM kp_relations WHERE src_id = ? AND rel_type = 'causal' AND dst_id = ?",
            (kp_root["id"], kp_target["id"]),
        )
        assert (await cur.fetchone())["cnt"] == 1

    @pytest.mark.asyncio
    async def test_rest_api_duplicate_relation_post_idempotency(self, challenger_app):
        transport = ASGITransport(app=challenger_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create two knowledge points
            r1 = await client.post("/api/courses/c_test/knowledge", json={"name": "Node Alpha"})
            r2 = await client.post("/api/courses/c_test/knowledge", json={"name": "Node Beta"})
            id_a, id_b = r1.json()["id"], r2.json()["id"]

            payload = {
                "src_id": id_a,
                "rel_type": "similar",
                "dst_id": id_b,
                "metadata": {"similarity_score": 0.88},
            }

            # Post 5 times identically
            for _ in range(5):
                resp = await client.post("/api/courses/c_test/knowledge/relations", json=payload)
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "ok"
                assert data["created"] is True

            # Query relations count
            r_get = await client.get(f"/api/courses/c_test/knowledge/{id_a}/relations")
            assert r_get.status_code == 200
            relations = r_get.json()["relations"]
            assert len(relations) == 1
            assert relations[0]["metadata"]["similarity_score"] == 0.88


# ============================================================================
# 2. Self-Loop Relations (src_id == dst_id)
# ============================================================================

class TestSelfLoopRelations:
    """Stress-test handling of self-loop edges across all ingestion and query pathways."""

    @pytest.mark.asyncio
    async def test_upsert_knowledge_point_filters_self_prereq(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        kp = await store.upsert_knowledge_point(
            course_id="c_self",
            name="自环测试节点",
            kp_id="kp_self_1",
            prereq_ids=["kp_self_1"],  # Self-loop prerequisite
        )
        assert kp["id"] == "kp_self_1"

        # Check kp_prereq: self-loop must not be inserted
        cur = await challenger_sqlite.conn.execute(
            "SELECT count(*) as cnt FROM kp_prereq WHERE kp_id = 'kp_self_1' AND prereq_id = 'kp_self_1'"
        )
        assert (await cur.fetchone())["cnt"] == 0

        # Check kp_relations: self-loop prerequisite must not be inserted
        cur2 = await challenger_sqlite.conn.execute(
            "SELECT count(*) as cnt FROM kp_relations WHERE src_id = 'kp_self_1' AND dst_id = 'kp_self_1'"
        )
        assert (await cur2.fetchone())["cnt"] == 0

    @pytest.mark.asyncio
    async def test_upsert_knowledge_point_filters_self_relations_batch(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        kp = await store.upsert_knowledge_point(
            course_id="c_self",
            name="自环批处理节点",
            kp_id="kp_self_2",
            relations=[
                {"src_id": "kp_self_2", "dst_id": "kp_self_2", "rel_type": "causal"},
                {"src_id": "kp_self_2", "dst_id": "kp_self_2", "rel_type": "similar"},
            ],
        )
        assert kp["id"] == "kp_self_2"

        # Verify no self-loops were written
        rels = await store.get_relations("kp_self_2")
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_direct_upsert_relation_self_loop_behavior_and_safe_deletion(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        # Directly call upsert_relation with self-loop
        rel = await store.upsert_relation(
            src_id="kp_isolated",
            rel_type="similar",
            dst_id="kp_isolated",
            metadata={"self_reference": True},
        )
        assert rel["src_id"] == "kp_isolated"
        assert rel["dst_id"] == "kp_isolated"

        # Query should safely return the relation without hanging or infinite loop
        rels_out = await store.get_relations("kp_isolated", direction="out")
        rels_in = await store.get_relations("kp_isolated", direction="in")
        rels_all = await store.get_relations("kp_isolated", direction="all")

        assert len(rels_out) == 1
        assert len(rels_in) == 1
        assert len(rels_all) == 1

        # Delete the self-loop relation cleanly
        deleted = await store.delete_relation("kp_isolated", "similar", "kp_isolated")
        assert deleted is True

        # Query again -> should be empty
        assert len(await store.get_relations("kp_isolated")) == 0

    @pytest.mark.asyncio
    async def test_rest_api_self_loop_roundtrip(self, challenger_app):
        transport = ASGITransport(app=challenger_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a node
            r_kp = await client.post("/api/courses/c_loop/knowledge", json={"name": "Node With Self Edge"})
            kp_id = r_kp.json()["id"]

            # Add self-loop via REST API
            r_edge = await client.post(
                "/api/courses/c_loop/knowledge/relations",
                json={
                    "src_id": kp_id,
                    "rel_type": "similar",
                    "dst_id": kp_id,
                    "metadata": {"type": "reflexive"},
                },
            )
            assert r_edge.status_code == 200
            assert r_edge.json()["created"] is True

            # Query relations
            r_list = await client.get(f"/api/courses/c_loop/knowledge/{kp_id}/relations")
            assert r_list.status_code == 200
            edges = r_list.json()["relations"]
            assert len(edges) == 1
            assert edges[0]["src_id"] == kp_id
            assert edges[0]["dst_id"] == kp_id

            # Delete self-loop
            r_del = await client.delete(
                "/api/courses/c_loop/knowledge/relations",
                params={"src_id": kp_id, "rel_type": "similar", "dst_id": kp_id},
            )
            assert r_del.status_code == 200
            assert r_del.json()["deleted"] is True


# ============================================================================
# 3. Non-Existent Knowledge Point IDs in Relations
# ============================================================================

class TestNonExistentKpIdsInRelations:
    """Stress-test system behavior when relation edges reference dangling or uncreated IDs."""

    @pytest.mark.asyncio
    async def test_insert_relation_with_dangling_ids_persists_without_foreign_key_crash(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        ghost_src = "kp_ghost_src_999"
        ghost_dst = "kp_ghost_dst_888"

        # Should insert cleanly (loose coupling enables distributed/offline canvas edge sync)
        rel = await store.upsert_relation(
            src_id=ghost_src,
            rel_type="causal",
            dst_id=ghost_dst,
            metadata={"note": "unanchored edge"},
        )
        assert rel["src_id"] == ghost_src
        assert rel["dst_id"] == ghost_dst

        # Querying by dangling ID must succeed and return the record
        ghost_rels = await store.get_relations(ghost_src)
        assert len(ghost_rels) == 1
        assert ghost_rels[0]["dst_id"] == ghost_dst

        # Course-scoped listing should exclude dangling relations whose IDs don't belong to the course
        course_rels = await store.list_course_relations("course_real")
        assert len(course_rels) == 0

        # Unscoped listing returns all relations including dangling
        all_rels = await store.list_course_relations(None)
        assert any(r["src_id"] == ghost_src for r in all_rels)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_relation_returns_false_safely(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        deleted = await store.delete_relation("kp_nonexistent_1", "causal", "kp_nonexistent_2")
        assert deleted is False

        deleted_by_id = await store.delete_relation("rel_phantom_id")
        assert deleted_by_id is False

    @pytest.mark.asyncio
    async def test_upsert_knowledge_point_with_dangling_prereq_ids(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        # Point to prereqs that do not exist yet in knowledge_points table
        kp = await store.upsert_knowledge_point(
            course_id="cs_async",
            name="异步计算",
            prereq_ids=["kp_future_1", "kp_future_2"],
        )
        assert kp["id"] is not None

        # Verify kp_prereq recorded them
        cur = await challenger_sqlite.conn.execute(
            "SELECT prereq_id FROM kp_prereq WHERE kp_id = ? ORDER BY prereq_id ASC",
            (kp["id"],),
        )
        rows = await cur.fetchall()
        assert [r["prereq_id"] for r in rows] == ["kp_future_1", "kp_future_2"]

        # When the future knowledge point is created later, it connects naturally
        kp_fut1 = await store.upsert_knowledge_point(
            course_id="cs_async",
            name="线程与进程",
            kp_id="kp_future_1",
        )
        assert kp_fut1["id"] == "kp_future_1"

        # Verify get_knowledge_point retrieves prereq_ids without error
        fetched = await store.get_knowledge_point(kp["id"])
        assert "kp_future_1" in fetched["prereq_ids"]
        assert "kp_future_2" in fetched["prereq_ids"]


# ============================================================================
# 4. Circular Dependency Graphs (A -> B -> C -> A)
# ============================================================================

class TestCircularDependencyGraphs:
    """Stress-test cyclic relation topologies to ensure all store and API query methods terminate safely."""

    @pytest.mark.asyncio
    async def test_three_node_cycle_prerequisite_queries_terminate_immediately(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        # Create 3 knowledge points: A -> B -> C -> A
        await store.upsert_knowledge_point("cycle_course", "概念 A", kp_id="cycle_a")
        await store.upsert_knowledge_point("cycle_course", "概念 B", kp_id="cycle_b")
        await store.upsert_knowledge_point("cycle_course", "概念 C", kp_id="cycle_c")

        # Establish circular prerequisite dependency: A -> B -> C -> A
        await store.upsert_relation("cycle_a", "prerequisite", "cycle_b")
        await store.upsert_relation("cycle_b", "prerequisite", "cycle_c")
        await store.upsert_relation("cycle_c", "prerequisite", "cycle_a")

        # Measure query execution times: must be instantaneous (O(1) table lookups, no unconstrained recursion)
        t0 = time.perf_counter()

        rels_a = await store.get_relations("cycle_a")
        rels_b = await store.get_relations("cycle_b")
        rels_c = await store.get_relations("cycle_c")

        kps_course = await store.list_course_relations("cycle_course")
        kp_a_details = await store.get_knowledge_point("cycle_a")

        elapsed = time.perf_counter() - t0
        assert elapsed < 0.2  # Must execute in under 200ms

        # Assert correct topology
        assert len(rels_a) == 2  # 1 in ('cycle_c'), 1 out ('cycle_b')
        assert len(rels_b) == 2  # 1 in ('cycle_a'), 1 out ('cycle_c')
        assert len(rels_c) == 2  # 1 in ('cycle_b'), 1 out ('cycle_a')
        assert len(kps_course) == 3
        assert "cycle_c" in kp_a_details["prereq_ids"]

    @pytest.mark.asyncio
    async def test_multi_type_cyclic_graph_dfs_oracle_traversal_with_cycle_guard(
        self, challenger_sqlite: SQLiteStore
    ):
        """Construct a 4-node complex cycle with different relation types:

        W --prerequisite--> X --causal--> Y --branch_followup--> Z --similar--> W
        Verify an empirical DFS traversal oracle detects the cycle and terminates cleanly.
        """
        store = LearningStore(challenger_sqlite)
        nodes = ["W", "X", "Y", "Z"]
        for n in nodes:
            await store.upsert_knowledge_point("c_mesh", f"Node {n}", kp_id=f"mesh_{n}")

        await store.upsert_relation("mesh_W", "prerequisite", "mesh_X")
        await store.upsert_relation("mesh_X", "causal", "mesh_Y")
        await store.upsert_relation("mesh_Y", "branch_followup", "mesh_Z")
        await store.upsert_relation("mesh_Z", "similar", "mesh_W")

        # Build in-memory adjacency list from database
        all_edges = await store.list_course_relations("c_mesh")
        adj: dict[str, list[str]] = {f"mesh_{n}": [] for n in nodes}
        for e in all_edges:
            adj[e["src_id"]].append(e["dst_id"])

        # DFS Cycle Detection Oracle with path guarding
        cycle_detected = False
        visited_nodes: list[str] = []

        def dfs(curr: str, current_path: set[str], max_depth: int = 10):
            nonlocal cycle_detected
            if max_depth <= 0:
                raise RecursionError("Infinite loop detected!")
            if curr in current_path:
                cycle_detected = True
                return
            visited_nodes.append(curr)
            for nxt in adj.get(curr, []):
                dfs(nxt, current_path | {curr}, max_depth - 1)

        dfs("mesh_W", set())
        assert cycle_detected is True
        assert "mesh_W" in visited_nodes
        assert len(visited_nodes) <= 10

    @pytest.mark.asyncio
    async def test_rest_api_cyclic_graph_endpoints_termination(self, challenger_app):
        transport = ASGITransport(app=challenger_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create two mutually dependent nodes: A <-> B
            r_a = await client.post("/api/courses/c_mutual/knowledge", json={"name": "Mutual A"})
            r_b = await client.post("/api/courses/c_mutual/knowledge", json={"name": "Mutual B"})
            id_a, id_b = r_a.json()["id"], r_b.json()["id"]

            await client.post(
                "/api/courses/c_mutual/knowledge/relations",
                json={"src_id": id_a, "rel_type": "similar", "dst_id": id_b},
            )
            await client.post(
                "/api/courses/c_mutual/knowledge/relations",
                json={"src_id": id_b, "rel_type": "similar", "dst_id": id_a},
            )

            # Query all endpoints
            r_a_rels = await client.get(f"/api/courses/c_mutual/knowledge/{id_a}/relations")
            assert r_a_rels.status_code == 200
            assert r_a_rels.json()["count"] == 2

            r_all = await client.get("/api/courses/c_mutual/knowledge/relations")
            assert r_all.status_code == 200
            assert r_all.json()["count"] == 2


# ============================================================================
# 5. Batch Inserts of 50+ Semantic Edges (Scale & Concurrency Stress)
# ============================================================================

class TestBatchInserts50PlusEdges:
    """Stress-test large-scale graph persistence with 80+ semantic relations and concurrent insertions."""

    @pytest.mark.asyncio
    async def test_bulk_insertion_of_80_relations_across_four_relation_types(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        course_id = "course_scale_stress"

        # Create 60 knowledge points
        kps = []
        for i in range(60):
            kp = await store.upsert_knowledge_point(
                course_id=course_id,
                name=f"大规模测试知识点_{i:03d}",
                kp_id=f"scale_kp_{i:03d}",
                summary=f"摘要描述 {i}",
            )
            kps.append(kp)

        assert len(kps) == 60

        # Construct 80 distinct semantic edges across all 4 relation types:
        # - 20 prerequisite edges: (0 -> 1, 1 -> 2, ..., 19 -> 20)
        # - 20 causal edges: (20 -> 21, ..., 39 -> 40)
        # - 20 similar edges: (40 -> 41, ..., 59 -> 0)
        # - 20 branch_followup edges: (0 -> 30, 1 -> 31, ..., 19 -> 49)
        relations_to_insert = []
        for i in range(20):
            relations_to_insert.append({
                "src_id": f"scale_kp_{i:03d}",
                "rel_type": "prerequisite",
                "dst_id": f"scale_kp_{(i+1):03d}",
                "metadata": {"edge_index": i, "category": "prereq"},
            })
        for i in range(20, 40):
            relations_to_insert.append({
                "src_id": f"scale_kp_{i:03d}",
                "rel_type": "causal",
                "dst_id": f"scale_kp_{(i+1):03d}",
                "metadata": {"edge_index": i, "category": "causal"},
            })
        for i in range(40, 60):
            relations_to_insert.append({
                "src_id": f"scale_kp_{i:03d}",
                "rel_type": "similar",
                "dst_id": f"scale_kp_{(i+1)%60:03d}",
                "metadata": {"edge_index": i, "category": "similar"},
            })
        for i in range(20):
            relations_to_insert.append({
                "src_id": f"scale_kp_{i:03d}",
                "rel_type": "branch_followup",
                "dst_id": f"scale_kp_{(i+30):03d}",
                "metadata": {"edge_index": i + 60, "category": "branch"},
            })

        assert len(relations_to_insert) == 80

        # Measure batch insertion performance
        t_start = time.perf_counter()
        for r in relations_to_insert:
            await store.upsert_relation(
                src_id=r["src_id"],
                rel_type=r["rel_type"],
                dst_id=r["dst_id"],
                metadata=r["metadata"],
            )
        duration = time.perf_counter() - t_start

        # 80 relations must insert in < 1.0 second on local SQLite
        assert duration < 1.0

        # Verify total course relations retrieved via SQL
        all_course_rels = await store.list_course_relations(course_id=course_id)
        assert len(all_course_rels) == 80

        # Verify filtering by rel_type
        prereqs = await store.list_course_relations(course_id=course_id, rel_type="prerequisite")
        causals = await store.list_course_relations(course_id=course_id, rel_type="causal")
        similars = await store.list_course_relations(course_id=course_id, rel_type="similar")
        branches = await store.list_course_relations(course_id=course_id, rel_type="branch_followup")

        assert len(prereqs) == 20
        assert len(causals) == 20
        assert len(similars) == 20
        assert len(branches) == 20

        # Verify dual-write to kp_prereq: exactly 20 prerequisite records
        cur_p = await challenger_sqlite.conn.execute("SELECT count(*) as cnt FROM kp_prereq")
        assert (await cur_p.fetchone())["cnt"] == 20

    @pytest.mark.asyncio
    async def test_concurrent_relation_upserts_no_deadlock_or_corruption(
        self, challenger_sqlite: SQLiteStore
    ):
        """Stress-test 20 concurrent asyncio tasks writing relations simultaneously."""
        store = LearningStore(challenger_sqlite)
        await store.upsert_knowledge_point("c_conc", "并发中心", kp_id="conc_hub")

        for i in range(20):
            await store.upsert_knowledge_point("c_conc", f"并发外围_{i}", kp_id=f"conc_spoke_{i}")

        async def worker(idx: int):
            return await store.upsert_relation(
                src_id="conc_hub",
                rel_type="causal" if idx % 2 == 0 else "similar",
                dst_id=f"conc_spoke_{idx}",
                metadata={"worker_idx": idx},
            )

        # Launch 20 concurrent coroutines
        results = await asyncio.gather(*(worker(i) for i in range(20)))
        assert len(results) == 20

        # Verify all 20 edges exist in database
        hub_edges = await store.get_relations("conc_hub", direction="out")
        assert len(hub_edges) == 20


# ============================================================================
# 6. Multi-Step Version History & Non-Destructive Rollback Stress
# ============================================================================

class TestVersionHistoryAndRollbackStress:
    """Stress-test revision timeline, history JSON integrity, and multi-step rollback chains."""

    @pytest.mark.asyncio
    async def test_seven_version_lifecycle_with_multiple_rollbacks(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        orig_quote = "不可变原始引文：微积分基本定理"

        # v1: Initial creation
        kp = await store.upsert_knowledge_point(
            course_id="c_history",
            name="微积分基本定理",
            raw_quote=orig_quote,
            content="版本 1 内容：牛顿-莱布尼茨公式初探。",
            provenance={"doc_title": "calculus.pdf", "page_start": 100},
        )
        kp_id = kp["id"]
        assert kp["version"] == 1
        assert kp["raw_quote"] == orig_quote

        # v2: User edits content
        await store.upsert_knowledge_point(
            course_id="c_history",
            name="微积分基本定理",
            content="版本 2 内容：引入变上限积分函数求导证明。",
            kp_id=kp_id,
        )

        # v3: User adds geometric notes
        await store.upsert_knowledge_point(
            course_id="c_history",
            name="微积分基本定理",
            content="版本 3 内容：结合面积与位移的几何直观。",
            kp_id=kp_id,
        )

        # Verify history has 3 snapshots
        revs = await store.get_revisions(kp_id)
        assert len(revs) == 3
        assert [r["version"] for r in revs] == [1, 2, 3]

        # Rollback 1: v3 -> rollback to v1 (creates v4)
        rb1 = await store.rollback_knowledge_point(kp_id, target_version=1)
        assert rb1["version"] == 4
        assert rb1["content"] == "版本 1 内容：牛顿-莱布尼茨公式初探。"
        assert rb1["raw_quote"] == orig_quote
        assert len(rb1["history"]) == 4
        assert rb1["history"][-1]["changeType"] == "rollback"

        # v5: User edits from rolled-back state
        await store.upsert_knowledge_point(
            course_id="c_history",
            name="微积分基本定理",
            content="版本 5 内容：在 v1 基础上做补充。",
            kp_id=kp_id,
        )

        # Rollback 2: v5 -> rollback to v2 (creates v6)
        rb2 = await store.rollback_knowledge_point(kp_id, target_version=2)
        assert rb2["version"] == 6
        assert rb2["content"] == "版本 2 内容：引入变上限积分函数求导证明。"
        assert rb2["raw_quote"] == orig_quote

        # Rollback 3: v6 -> rollback to v3 (creates v7)
        rb3 = await store.rollback_knowledge_point(kp_id, target_version=3)
        assert rb3["version"] == 7
        assert rb3["content"] == "版本 3 内容：结合面积与位移的几何直观。"
        assert rb3["raw_quote"] == orig_quote

        # Final check on persistent state
        final_kp = await store.get_knowledge_point(kp_id)
        assert final_kp["version"] == 7
        assert final_kp["raw_quote"] == orig_quote
        assert final_kp["provenance"]["page_start"] == 100
        assert len(final_kp["history"]) == 7


# ============================================================================
# 7. Mastery Auto-Enrollment SM-2 Persistence & Progress Protection
# ============================================================================

class TestMasteryAutoEnrollmentAndProtection:
    """Verify newly created knowledge points auto-enroll without resetting existing progress."""

    @pytest.mark.asyncio
    async def test_upsert_enrols_new_kp_and_protects_existing_mastery(
        self, challenger_sqlite: SQLiteStore
    ):
        store = LearningStore(challenger_sqlite)
        user_id = "student_challenger"

        # 1. Create a knowledge point
        kp = await store.upsert_knowledge_point(
            course_id="math_sm2",
            name="傅里叶级数",
            user_id=user_id,
        )
        kp_id = kp["id"]

        # Check initial mastery registration
        cur = await challenger_sqlite.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?",
            (user_id, kp_id),
        )
        row = await cur.fetchone()
        assert row is not None
        assert row["mastery"] == 0.3
        assert row["ef"] == 2.5
        assert row["repetitions"] == 0

        # 2. Simulate user studying and upgrading mastery to 0.85
        await challenger_sqlite.conn.execute(
            "UPDATE mastery SET mastery = 0.85, repetitions = 3, interval_days = 15.0 WHERE user_id = ? AND kp_id = ?",
            (user_id, kp_id),
        )
        await challenger_sqlite.conn.commit()

        # 3. Upsert/edit the same knowledge point again (e.g. content edit)
        await store.upsert_knowledge_point(
            course_id="math_sm2",
            name="傅里叶级数",
            kp_id=kp_id,
            content="补充正交性证明",
            user_id=user_id,
        )

        # 4. Verify mastery was NOT overwritten back to 0.3
        cur2 = await challenger_sqlite.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?",
            (user_id, kp_id),
        )
        row2 = await cur2.fetchone()
        assert row2["mastery"] == 0.85
        assert row2["repetitions"] == 3
        assert row2["interval_days"] == 15.0
