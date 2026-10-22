"""Empirical Challenger M2-2 Stress Test Suite.

Adversarially verifies:
1. Rapid multi-version edits (v1 -> v2 -> v3 -> v4).
2. Rollback to v1 (v4 -> v5 matching v1, with all 4 prior versions preserved in history_json).
3. Rollback of a rolled-back node (v5 -> rollback to v3 [v6], followed by rollback to v5 [v7], followed by normal edit [v8]).
4. Mastery retention: verify subsequent updates and rollbacks DO NOT overwrite or reset student EF, repetition count, interval, or mastery.
5. Large payloads (100KB+ to 250KB+ markdown notes) in content, history_json serialization, and REST API roundtrips.
6. Boundary and failure modes: invalid versions, nonexistent nodes, raw_quote immutability, provenance preservation.
"""

from __future__ import annotations

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
async def sqlite_store(tmp_path: Path):
    db_path = tmp_path / "test_m2_challenger.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    yield store
    await store.aclose()


@pytest.fixture
async def learning_store(sqlite_store: SQLiteStore):
    return LearningStore(sqlite_store)


@pytest.fixture
async def api_client_and_container(tmp_path: Path):
    db_path = tmp_path / "test_api_m2_challenger.db"
    settings = Settings(
        offline_mode=True,
        sqlite_path=db_path,
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, container, container.learning_store

    await container.aclose()


class TestMultiVersionEdits:
    """Stress tests rapid multi-version editing sequence v1 -> v2 -> v3 -> v4."""

    async def test_rapid_edits_and_version_chain(self, learning_store: LearningStore, sqlite_store: SQLiteStore):
        cid = "course_multi_ver"
        name = "梯度反向传播"
        initial_quote = "通过链式法则逐层回传误差梯度。"
        initial_content = "v1: 基础链式法则公式。"
        initial_prov = {
            "message_id": "msg_001",
            "doc_title": "DeepLearning.pdf",
            "page_start": 42,
            "page_end": 43,
        }

        # 1. 创建 v1
        kp = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            summary="v1 summary",
            raw_quote=initial_quote,
            content=initial_content,
            provenance=initial_prov,
            user_id="learner_alpha",
        )
        kp_id = kp["id"]
        assert kp["version"] == 1
        assert kp["content"] == initial_content
        assert kp["raw_quote"] == initial_quote
        assert len(kp["history"]) == 1
        assert kp["history"][0]["version"] == 1
        assert kp["history"][0]["content"] == initial_content

        # 2. 编辑到 v2: 尝试篡改 raw_quote（必须被保护忽略），更新 content
        v2_content = "v2: 增加向量化矩阵偏导推导 $\\frac{\\partial L}{\\partial W} = \\delta X^T$。"
        v2_res = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            summary="v2 summary",
            raw_quote="恶意篡改的原始引文",
            content=v2_content,
            kp_id=kp_id,
            user_id="learner_alpha",
        )
        assert v2_res["id"] == kp_id
        assert v2_res["version"] == 2
        assert v2_res["raw_quote"] == initial_quote, "raw_quote must be immutable once created!"
        assert v2_res["content"] == v2_content
        assert len(v2_res["history"]) == 2
        assert v2_res["history"][0]["version"] == 1
        assert v2_res["history"][1]["version"] == 2
        assert v2_res["history"][1]["content"] == v2_content

        # 3. 编辑到 v3: 包含复杂的数学公式、代码块、特殊字符
        v3_content = (
            "v3: Python 自动微分实现:\n```python\n"
            "def backward(grad_output):\n"
            "    return grad_output * self.weight.T\n"
            "```\n公式: $\\nabla_\\theta J(\\theta) = \\mathbb{E}[\\nabla_\\theta \\log \\pi_\\theta(s,a) Q(s,a)]$\n"
            "特殊字符: <script>alert('xss')</script> & \"quoted\" & emoji 🚀"
        )
        v3_res = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            content=v3_content,
            kp_id=kp_id,
            user_id="learner_alpha",
        )
        assert v3_res["version"] == 3
        assert v3_res["raw_quote"] == initial_quote
        assert v3_res["content"] == v3_content
        assert len(v3_res["history"]) == 3
        assert v3_res["history"][2]["version"] == 3

        # 4. 编辑到 v4: 包含空白行和结构化 Markdown 表格
        v4_content = (
            "v4: 性能对比表格:\n\n"
            "| 优化器 | 收敛速度 | 显存占用 |\n"
            "|---|---|---|\n"
            "| SGD | 慢 | 低 |\n"
            "| Adam | 快 | 高 |\n"
        )
        v4_res = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            content=v4_content,
            kp_id=kp_id,
            user_id="learner_alpha",
        )
        assert v4_res["version"] == 4
        assert v4_res["content"] == v4_content
        assert len(v4_res["history"]) == 4

        # 验证 SQLite 数据库持久化状态
        cur = await sqlite_store.conn.execute(
            "SELECT version, content, raw_quote, history_json FROM knowledge_points WHERE id = ?",
            (kp_id,),
        )
        row = await cur.fetchone()
        assert row is not None
        assert row["version"] == 4
        assert row["content"] == v4_content
        assert row["raw_quote"] == initial_quote
        history_from_db = json.loads(row["history_json"])
        assert len(history_from_db) == 4
        assert [h["version"] for h in history_from_db] == [1, 2, 3, 4]

    async def test_high_frequency_edit_loop(self, learning_store: LearningStore):
        """高频连续 20 次更新，确保版本号连续自增且 history_json 结构完好无损。"""
        cid = "course_stress_loop"
        kp = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name="高频更新测试",
            content="version 1 text",
        )
        kp_id = kp["id"]

        for i in range(2, 22):
            res = await learning_store.upsert_knowledge_point(
                course_id=cid,
                name="高频更新测试",
                content=f"version {i} text with timestamp {time.time()}",
                kp_id=kp_id,
            )
            assert res["version"] == i
            assert len(res["history"]) == i

        fetched = await learning_store.get_knowledge_point(kp_id)
        assert fetched["version"] == 21
        assert len(fetched["history"]) == 21
        assert fetched["history"][-1]["version"] == 21


class TestRollbackSemantics:
    """Stress tests non-destructive rollback to v1, rollback of a rolled-back node, and version ancestry."""

    async def test_rollback_to_v1_creates_v5_and_preserves_all_prior_history(
        self, learning_store: LearningStore, sqlite_store: SQLiteStore
    ):
        cid = "course_rollback_test"
        name = "泰勒余项公式"
        initial_quote = "Rn(x) = f^(n+1)(xi) / (n+1)! * (x - x0)^(n+1)"
        v1_text = "v1: 初代定义与皮亚诺余项"
        v2_text = "v2: 引入拉格朗日余项"
        v3_text = "v3: 引入积分型余项"
        v4_text = "v4: 引入柯西余项与多维泰勒级数"

        kp = await learning_store.upsert_knowledge_point(
            course_id=cid, name=name, raw_quote=initial_quote, content=v1_text
        )
        kp_id = kp["id"]
        await learning_store.upsert_knowledge_point(course_id=cid, name=name, content=v2_text, kp_id=kp_id)
        await learning_store.upsert_knowledge_point(course_id=cid, name=name, content=v3_text, kp_id=kp_id)
        await learning_store.upsert_knowledge_point(course_id=cid, name=name, content=v4_text, kp_id=kp_id)

        # 检查当前处于 v4
        revs_before = await learning_store.get_revisions(kp_id)
        assert len(revs_before) == 4
        assert [r["version"] for r in revs_before] == [1, 2, 3, 4]

        # 执行回滚至 v1 -> 必须生成 v5
        v5_res = await learning_store.rollback_knowledge_point(kp_id, target_version=1)
        assert v5_res["id"] == kp_id
        assert v5_res["version"] == 5, "Rollback must produce incremented version v5, not overwrite v1!"
        assert v5_res["content"] == v1_text, "v5 content must match v1 content exactly!"
        assert v5_res["raw_quote"] == initial_quote

        # 验证历史数组包含全部 5 个版本，且前 4 个版本原封不动
        history = v5_res["history"]
        assert len(history) == 5, "history must now contain 5 records (v1..v5)"
        assert history[0]["version"] == 1 and history[0]["content"] == v1_text
        assert history[1]["version"] == 2 and history[1]["content"] == v2_text
        assert history[2]["version"] == 3 and history[2]["content"] == v3_text
        assert history[3]["version"] == 4 and history[3]["content"] == v4_text
        assert history[4]["version"] == 5 and history[4]["content"] == v1_text
        assert history[4]["target_version"] == 1
        assert history[4]["changeType"] == "rollback"

        # 验证直接从 SQLite 查询
        cur = await sqlite_store.conn.execute(
            "SELECT version, content, history_json FROM knowledge_points WHERE id = ?", (kp_id,)
        )
        row = await cur.fetchone()
        assert row["version"] == 5
        assert row["content"] == v1_text
        db_history = json.loads(row["history_json"])
        assert len(db_history) == 5
        assert [h["version"] for h in db_history] == [1, 2, 3, 4, 5]

    async def test_rollback_of_rolled_back_node_v5_to_v3_creates_v6(
        self, learning_store: LearningStore
    ):
        """测试对已回滚节点再次回滚（v5 -> 回滚至 v3 [v6]），以及二次回滚至 v5 [v7]，再继续正常编辑 [v8]。"""
        cid = "course_multi_rollback"
        name = "傅里叶变换"
        kp = await learning_store.upsert_knowledge_point(course_id=cid, name=name, content="v1: 傅里叶级数")
        kp_id = kp["id"]
        await learning_store.upsert_knowledge_point(course_id=cid, name=name, content="v2: 离散傅里叶变换 (DFT)", kp_id=kp_id)
        await learning_store.upsert_knowledge_point(course_id=cid, name=name, content="v3: 快速傅里叶变换 (FFT)", kp_id=kp_id)
        await learning_store.upsert_knowledge_point(course_id=cid, name=name, content="v4: 小波变换对比", kp_id=kp_id)

        # 步骤 1: 回滚到 v1 -> 生成 v5
        v5 = await learning_store.rollback_knowledge_point(kp_id, target_version=1)
        assert v5["version"] == 5
        assert v5["content"] == "v1: 傅里叶级数"

        # 步骤 2: 在 v5 的基础上回滚到 v3 -> 必须生成 v6，内容匹配 v3
        v6 = await learning_store.rollback_knowledge_point(kp_id, target_version=3)
        assert v6["version"] == 6
        assert v6["content"] == "v3: 快速傅里叶变换 (FFT)"
        assert len(v6["history"]) == 6
        assert [h["version"] for h in v6["history"]] == [1, 2, 3, 4, 5, 6]
        assert v6["history"][5]["target_version"] == 3

        # 步骤 3: 在 v6 的基础上回滚到 v5（回滚到之前生成的 rollback 节点）-> 必须生成 v7
        v7 = await learning_store.rollback_knowledge_point(kp_id, target_version=5)
        assert v7["version"] == 7
        assert v7["content"] == "v1: 傅里叶级数"  # 因为 v5 的内容就是 v1
        assert len(v7["history"]) == 7
        assert v7["history"][6]["target_version"] == 5

        # 步骤 4: 在多次回滚后继续执行正常编辑更新 -> 必须平滑递增至 v8
        v8 = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            content="v8: 最新更新，加入分数阶傅里叶变换",
            kp_id=kp_id,
        )
        assert v8["version"] == 8
        assert v8["content"] == "v8: 最新更新，加入分数阶傅里叶变换"
        assert len(v8["history"]) == 8
        assert v8["history"][7]["changeType"] == "edit"

    async def test_rollback_invalid_boundaries(self, learning_store: LearningStore):
        """验证非法版本回滚（不存在版本、负数、空节点等）正确抛出异常。"""
        kp = await learning_store.upsert_knowledge_point(
            course_id="c_err", name="测试边界", content="initial content"
        )
        kp_id = kp["id"]

        with pytest.raises(ValueError, match="Target version 99 does not exist"):
            await learning_store.rollback_knowledge_point(kp_id, target_version=99)

        with pytest.raises(ValueError, match="Target version 0 does not exist"):
            await learning_store.rollback_knowledge_point(kp_id, target_version=0)

        with pytest.raises(ValueError, match="Target version -1 does not exist"):
            await learning_store.rollback_knowledge_point(kp_id, target_version=-1)

        with pytest.raises(KeyError, match="not found"):
            await learning_store.rollback_knowledge_point("kp_nonexistent_xyz", target_version=1)

    async def test_rest_api_full_revision_and_multi_rollback_lifecycle(self, api_client_and_container):
        """通过 REST API 端到端验证连续编辑 (v1->v2->v3->v4)、回滚至 v1 (v5)、回滚至 v3 (v6) 及历史时间轴。"""
        client, container, store = api_client_and_container
        cid = "course_rest_lifecycle"

        # 1. POST 创建 v1
        r1 = await client.post(
            f"/api/courses/{cid}/knowledge",
            json={
                "name": "K-Means 聚类",
                "summary": "无监督质心迭代算法",
                "raw_quote": "通过迭代更新质心最小化簇内方差和。",
                "content": "v1: 标准 Lloyd 算法步骤。",
            },
        )
        assert r1.status_code == 200
        kp_id = r1.json()["id"]
        assert r1.json()["version"] == 1

        # 2. 连续编辑到 v2, v3, v4
        r2 = await client.post(f"/api/courses/{cid}/knowledge", json={"id": kp_id, "name": "K-Means 聚类", "content": "v2: K-Means++ 初始化策略。"})
        assert r2.status_code == 200 and r2.json()["version"] == 2

        r3 = await client.post(f"/api/courses/{cid}/knowledge", json={"id": kp_id, "name": "K-Means 聚类", "content": "v3: Mini-Batch K-Means 大规模优化。"})
        assert r3.status_code == 200 and r3.json()["version"] == 3

        r4 = await client.post(f"/api/courses/{cid}/knowledge", json={"id": kp_id, "name": "K-Means 聚类", "content": "v4: 肘部法与轮廓系数评估指标。"})
        assert r4.status_code == 200 and r4.json()["version"] == 4

        # 3. 回滚至 v1 -> 生成 v5
        rb1 = await client.post(f"/api/courses/{cid}/knowledge/{kp_id}/rollback", json={"target_version": 1})
        assert rb1.status_code == 200
        res_rb1 = rb1.json()
        assert res_rb1["version"] == 5
        assert res_rb1["rolled_back_to"] == 1
        assert res_rb1["knowledge_point"]["content"] == "v1: 标准 Lloyd 算法步骤。"

        # 4. 回滚至 v3 -> 生成 v6
        rb2 = await client.post(f"/api/courses/{cid}/knowledge/{kp_id}/rollback", json={"target_version": 3})
        assert rb2.status_code == 200
        res_rb2 = rb2.json()
        assert res_rb2["version"] == 6
        assert res_rb2["rolled_back_to"] == 3
        assert res_rb2["knowledge_point"]["content"] == "v3: Mini-Batch K-Means 大规模优化。"

        # 5. 查询修订历史列表 (GET /api/courses/{course_id}/knowledge/{kp_id}/revisions)
        revs_resp = await client.get(f"/api/courses/{cid}/knowledge/{kp_id}/revisions")
        assert revs_resp.status_code == 200
        revs_data = revs_resp.json()
        assert revs_data["current_version"] == 6
        assert revs_data["count"] == 6
        revisions = revs_data["revisions"]
        assert [r["version"] for r in revisions] == [1, 2, 3, 4, 5, 6]
        assert revisions[0]["content"] == "v1: 标准 Lloyd 算法步骤。"
        assert revisions[2]["content"] == "v3: Mini-Batch K-Means 大规模优化。"
        assert revisions[4]["content"] == "v1: 标准 Lloyd 算法步骤。"
        assert revisions[5]["content"] == "v3: Mini-Batch K-Means 大规模优化。"


class TestMasteryRetention:
    """Stress tests mastery retention: subsequent updates or rollbacks DO NOT wipe student EF or repetitions."""

    async def test_mastery_retention_across_updates_and_rollbacks(
        self, learning_store: LearningStore, sqlite_store: SQLiteStore
    ):
        user_id = "student_diligent_01"
        cid = "course_ai"
        name = "注意力机制与 Transformer"

        # 1. 创建知识点 -> 自动在 mastery 表中登记初始状态
        kp = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            summary="Self-Attention 概念",
            content="v1: Scaled Dot-Product Attention",
            user_id=user_id,
        )
        kp_id = kp["id"]

        # 检查初始掌握度
        cur = await sqlite_store.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?", (user_id, kp_id)
        )
        m_init = await cur.fetchone()
        assert m_init is not None
        assert m_init["ef"] == 2.5
        assert m_init["repetitions"] == 0
        assert m_init["interval_days"] == 1.0
        assert m_init["mastery"] == 0.3

        # 2. 学生做题学习：连续作答 3 次，使 EF、repetitions、interval 和 mastery 得到提升
        q = await learning_store.add_question(
            kp_id=kp_id,
            q_type="choice",
            stem="Attention 机制的缩放因子是？",
            answer="sqrt(d_k)",
        )
        qid = q["id"]

        # 作答 1: 答对
        att1 = await learning_store.record_attempt_and_update_mastery(
            user_id=user_id, question_id=qid, user_answer="sqrt(d_k)", is_correct=True, score=1.0
        )
        assert att1["repetitions"] == 1
        assert att1["new_mastery"] > 0.3

        # 作答 2: 答对
        att2 = await learning_store.record_attempt_and_update_mastery(
            user_id=user_id, question_id=qid, user_answer="sqrt(d_k)", is_correct=True, score=1.0
        )
        assert att2["repetitions"] == 2
        assert att2["interval_days"] == 6.0

        # 作答 3: 答对
        att3 = await learning_store.record_attempt_and_update_mastery(
            user_id=user_id, question_id=qid, user_answer="sqrt(d_k)", is_correct=True, score=1.0
        )
        assert att3["repetitions"] == 3
        assert att3["interval_days"] > 6.0

        # 获取学生当前的学术成就记录
        cur = await sqlite_store.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?", (user_id, kp_id)
        )
        m_earned = await cur.fetchone()
        earned_ef = float(m_earned["ef"])
        earned_reps = int(m_earned["repetitions"])
        earned_interval = float(m_earned["interval_days"])
        earned_mastery = float(m_earned["mastery"])
        earned_next_review = float(m_earned["next_review_at"])

        assert earned_ef >= 2.5
        assert earned_reps == 3
        assert earned_interval > 6.0
        assert earned_mastery > 0.5

        # 3. 对该知识点进行多次更新（编辑正文、调整难度、更新摘要）
        for edit_idx in range(2, 6):
            await learning_store.upsert_knowledge_point(
                course_id=cid,
                name=name,
                summary=f"更新摘要 v{edit_idx}",
                difficulty=0.8,
                content=f"v{edit_idx}: 补充 Multi-Head 注意力与交叉注意力细节",
                kp_id=kp_id,
                user_id=user_id,
            )
            # 严苛验证：每一次更新后，mastery 表绝对不能被 reset 到 0.3 / 2.5 / 0
            cur_check = await sqlite_store.conn.execute(
                "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?", (user_id, kp_id)
            )
            m_now = await cur_check.fetchone()
            assert float(m_now["ef"]) == earned_ef, f"EF was reset after edit v{edit_idx}!"
            assert int(m_now["repetitions"]) == earned_reps, f"Repetitions was reset after edit v{edit_idx}!"
            assert float(m_now["interval_days"]) == earned_interval, f"Interval was reset after edit v{edit_idx}!"
            assert float(m_now["mastery"]) == earned_mastery, f"Mastery was reset after edit v{edit_idx}!"
            assert float(m_now["next_review_at"]) == earned_next_review

        # 4. 执行回滚操作（回滚到 v1 生成 v6，再回滚到 v3 生成 v7）
        await learning_store.rollback_knowledge_point(kp_id, target_version=1, user_id=user_id)
        await learning_store.rollback_knowledge_point(kp_id, target_version=3, user_id=user_id)

        cur_after_rollback = await sqlite_store.conn.execute(
            "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?", (user_id, kp_id)
        )
        m_final = await cur_after_rollback.fetchone()
        assert float(m_final["ef"]) == earned_ef, "EF was overwritten during rollback!"
        assert int(m_final["repetitions"]) == earned_reps, "Repetitions was overwritten during rollback!"
        assert float(m_final["interval_days"]) == earned_interval, "Interval was overwritten during rollback!"
        assert float(m_final["mastery"]) == earned_mastery, "Mastery was overwritten during rollback!"
        assert float(m_final["next_review_at"]) == earned_next_review

    async def test_multi_student_mastery_isolation_and_retention(
        self, learning_store: LearningStore, sqlite_store: SQLiteStore
    ):
        """验证多名学生在同一知识点上的异构掌握度记录在知识点升级和回滚时均能无损隔离并保留。"""
        cid = "course_multi_user"
        name = "多头注意力机制"

        kp = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            content="v1 初版",
            user_id="student_alice",
        )
        kp_id = kp["id"]

        # 手工为不同学生模拟不同的学习进度
        now = time.time()
        # Alice: 高掌握度
        await sqlite_store.conn.execute(
            """
            INSERT INTO mastery (user_id, kp_id, mastery, ef, interval_days, repetitions, next_review_at, updated_at)
            VALUES ('alice', ?, 0.92, 2.8, 30.0, 5, ?, ?)
            ON CONFLICT(user_id, kp_id) DO UPDATE SET mastery=0.92, ef=2.8, interval_days=30.0, repetitions=5
            """,
            (kp_id, now + 86400 * 30, now),
        )
        # Bob: 初学状态
        await sqlite_store.conn.execute(
            """
            INSERT INTO mastery (user_id, kp_id, mastery, ef, interval_days, repetitions, next_review_at, updated_at)
            VALUES ('bob', ?, 0.45, 2.3, 2.0, 1, ?, ?)
            ON CONFLICT(user_id, kp_id) DO UPDATE SET mastery=0.45, ef=2.3, interval_days=2.0, repetitions=1
            """,
            (kp_id, now + 86400 * 2, now),
        )
        # Charlie: 默认初始状态
        await sqlite_store.conn.execute(
            """
            INSERT INTO mastery (user_id, kp_id, mastery, ef, interval_days, repetitions, next_review_at, updated_at)
            VALUES ('charlie', ?, 0.3, 2.5, 1.0, 0, ?, ?)
            ON CONFLICT(user_id, kp_id) DO NOTHING
            """,
            (kp_id, now, now),
        )
        await sqlite_store.conn.commit()

        # 触发知识点多次修订与回滚
        await learning_store.upsert_knowledge_point(
            course_id=cid, name=name, content="v2 更新", kp_id=kp_id, user_id="student_alice"
        )
        await learning_store.upsert_knowledge_point(
            course_id=cid, name=name, content="v3 再次更新", kp_id=kp_id, user_id="student_alice"
        )
        await learning_store.rollback_knowledge_point(
            kp_id, target_version=1, user_id="student_alice"
        )

        # 逐一检查 Alice, Bob, Charlie 的掌握度数据
        cur = await sqlite_store.conn.execute("SELECT * FROM mastery WHERE kp_id = ?", (kp_id,))
        rows = {r["user_id"]: dict(r) for r in await cur.fetchall()}

        assert rows["alice"]["mastery"] == 0.92
        assert rows["alice"]["ef"] == 2.8
        assert rows["alice"]["repetitions"] == 5

        assert rows["bob"]["mastery"] == 0.45
        assert rows["bob"]["ef"] == 2.3
        assert rows["bob"]["repetitions"] == 1

        assert rows["charlie"]["mastery"] == 0.3
        assert rows["charlie"]["ef"] == 2.5
        assert rows["charlie"]["repetitions"] == 0


class TestLargePayloadSerialization:
    """Stress tests 100KB+ markdown notes, JSON serialization, and SQLite TEXT stability."""

    def _generate_large_markdown(self, target_kb: int) -> str:
        """Generates realistic structured markdown content with headers, formulas, and text."""
        paragraph = (
            "### 3.1 深度卷积神经网络与注意力机制融合\n"
            "在现代计算机视觉任务中，Transformer 与 CNN 的结合展示出了极高的表征学习效率。\n"
            "对于输入特征图 $\\mathbf{X} \\in \\mathbb{R}^{H \\times W \\times C}$，"
            "其局部卷积算子可表示为：\n"
            "$$\\mathbf{Y}_{i,j} = \\sum_{m=-k}^k \\sum_{n=-k}^k \\mathbf{W}_{m,n} \\mathbf{X}_{i+m, j+n} + \\mathbf{b}$$\n"
            "代码段示例：\n"
            "```python\n"
            "class ConvBlock(nn.Module):\n"
            "    def __init__(self, in_c, out_c):\n"
            "        super().__init__()\n"
            "        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)\n"
            "        self.bn = nn.BatchNorm2d(out_c)\n"
            "        self.relu = nn.ReLU(inplace=True)\n"
            "    def forward(self, x):\n"
            "        return self.relu(self.bn(self.conv(x)))\n"
            "```\n"
            "| 特征维度 | 浮点运算量 (FLOPs) | 参数量 (Params) | 准确率 (Top-1 Acc) |\n"
            "|---|---|---|---|\n"
            "| 256 | 4.2 GFLOPs | 25.6 M | 82.4% |\n"
            "| 512 | 8.5 GFLOPs | 51.2 M | 84.1% |\n\n"
        )
        repeats = (target_kb * 1024) // len(paragraph.encode("utf-8")) + 1
        return paragraph * repeats

    async def test_100kb_and_250kb_payload_serialization(
        self, learning_store: LearningStore, sqlite_store: SQLiteStore
    ):
        cid = "course_large_payload"
        name = "计算机视觉大模型综述"
        large_v1 = self._generate_large_markdown(120)  # ~120KB
        payload_bytes_v1 = len(large_v1.encode("utf-8"))
        assert payload_bytes_v1 >= 100 * 1024, f"Payload size {payload_bytes_v1} is less than 100KB"

        # 1. 写入 120KB 初始笔记
        kp = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            summary="120KB 深度综述",
            content=large_v1,
        )
        kp_id = kp["id"]
        assert kp["version"] == 1
        assert len(kp["content"]) == len(large_v1)

        # 2. 读取并验证 SQLite 字节保真度
        fetched = await learning_store.get_knowledge_point(kp_id)
        assert fetched is not None
        assert fetched["content"] == large_v1

        # 3. 更新为 250KB 笔记 (v2)
        large_v2 = self._generate_large_markdown(250)  # ~250KB
        payload_bytes_v2 = len(large_v2.encode("utf-8"))
        assert payload_bytes_v2 >= 250 * 1024

        v2 = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name=name,
            summary="250KB 扩展综述",
            content=large_v2,
            kp_id=kp_id,
        )
        assert v2["version"] == 2
        assert len(v2["content"]) == len(large_v2)

        # 4. 回滚至 120KB 的 v1 -> 生成 v3
        v3 = await learning_store.rollback_knowledge_point(kp_id, target_version=1)
        assert v3["version"] == 3
        assert v3["content"] == large_v1
        assert len(v3["history"]) == 3

        # 5. 获取历史版本时间轴并反序列化全部大快照
        revs = await learning_store.get_revisions(kp_id)
        assert len(revs) == 3
        assert revs[0]["content"] == large_v1
        assert revs[1]["content"] == large_v2
        assert revs[2]["content"] == large_v1

        # 验证 SQLite 直接存储结构
        cur = await sqlite_store.conn.execute("SELECT LENGTH(history_json) as h_len FROM knowledge_points WHERE id = ?", (kp_id,))
        row = await cur.fetchone()
        assert row["h_len"] > 300 * 1024, "history_json should easily exceed 300KB with multiple large revisions"

    async def test_rest_api_large_payload_and_rollback_roundtrip(self, api_client_and_container):
        """通过 FastAPI REST 接口验证 120KB+ 载荷的创建、修订、历史查询与回滚往返。"""
        client, container, store = api_client_and_container
        large_content_1 = self._generate_large_markdown(120)
        large_content_2 = self._generate_large_markdown(140)

        # 1. POST /api/courses/{course_id}/knowledge
        r1 = await client.post(
            "/api/courses/course_web_large/knowledge",
            json={
                "name": "分布式系统与共识算法",
                "summary": "Paxos 与 Raft 算法全景",
                "content": large_content_1,
                "provenance": {
                    "doc_title": "distributed_systems.pdf",
                    "page_start": 100,
                    "page_end": 180,
                },
            },
        )
        assert r1.status_code == 200, r1.text
        kp_id = r1.json()["id"]
        assert r1.json()["version"] == 1

        # 2. 更新为 v2
        r2 = await client.post(
            "/api/courses/course_web_large/knowledge",
            json={
                "id": kp_id,
                "name": "分布式系统与共识算法",
                "content": large_content_2,
            },
        )
        assert r2.status_code == 200
        assert r2.json()["version"] == 2

        # 3. GET /api/courses/{course_id}/knowledge/{kp_id}/revisions
        r3 = await client.get(f"/api/courses/course_web_large/knowledge/{kp_id}/revisions")
        assert r3.status_code == 200
        revs = r3.json()["revisions"]
        assert len(revs) == 2
        assert len(revs[0]["content"]) == len(large_content_1)
        assert len(revs[1]["content"]) == len(large_content_2)

        # 4. POST /api/courses/{course_id}/knowledge/{kp_id}/rollback
        r4 = await client.post(
            f"/api/courses/course_web_large/knowledge/{kp_id}/rollback",
            json={"target_version": 1},
        )
        assert r4.status_code == 200
        res4 = r4.json()
        assert res4["version"] == 3
        assert res4["knowledge_point"]["content"] == large_content_1


class TestDualZoneAndProvenanceProtection:
    """Stress tests immutable quote area and provenance persistence."""

    async def test_provenance_retention_when_updating_notes(self, learning_store: LearningStore):
        cid = "course_prov"
        prov = {
            "message_id": "msg_anchor_99",
            "turn_index": 5,
            "doc_title": "QuantumComputing.pdf",
            "page_start": 77,
            "page_end": 79,
            "quote_snippet": "量子比特可以处于 |0> 和 |1> 的叠加态。",
        }

        # 创建节点
        kp = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name="量子叠加态",
            raw_quote=prov["quote_snippet"],
            content="初版笔记",
            provenance=prov,
        )
        kp_id = kp["id"]

        # 后续仅更新 content，不传 provenance
        kp_updated = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name="量子叠加态",
            content="第二版：布洛赫球面表示 $|\\psi\\rangle = \\cos(\\theta/2)|0\\rangle + e^{i\\phi}\\sin(\\theta/2)|1\\rangle$",
            kp_id=kp_id,
        )

        assert kp_updated["version"] == 2
        # provenance 必须完整保留，不得被冲掉
        assert kp_updated["provenance"]["message_id"] == "msg_anchor_99"
        assert kp_updated["provenance"]["doc_title"] == "QuantumComputing.pdf"
        assert kp_updated["provenance"]["page_start"] == 77
        assert kp_updated["provenance"]["page_end"] == 79
        assert kp_updated["raw_quote"] == prov["quote_snippet"]

        # 验证 get_knowledge_point 也能读取完整 provenance
        fetched = await learning_store.get_knowledge_point(kp_id)
        assert fetched["provenance"]["page_start"] == 77

    async def test_late_raw_quote_locking(self, learning_store: LearningStore):
        """测试初始未传 raw_quote 时首次补充生效，后续任何修改均被锁定的安全策略。"""
        cid = "course_lock"
        # 1. 创建节点时 raw_quote 为空
        kp1 = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name="图卷积神经网络 (GCN)",
            content="初始笔记",
            raw_quote="",
        )
        kp_id = kp1["id"]
        assert kp1["raw_quote"] == ""

        # 2. 第一次补充 raw_quote -> 允许写入
        kp2 = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name="图卷积神经网络 (GCN)",
            content="更新笔记",
            raw_quote="首次补充的原始引文：H^(l+1) = sigma(D~^(-1/2) A~ D~^(-1/2) H^(l) W^(l))",
            kp_id=kp_id,
        )
        assert kp2["version"] == 2
        assert "首次补充的原始引文" in kp2["raw_quote"]

        # 3. 后续再次尝试覆盖 raw_quote -> 必须被拒绝/忽略，锁定原有引文
        kp3 = await learning_store.upsert_knowledge_point(
            course_id=cid,
            name="图卷积神经网络 (GCN)",
            content="再次更新笔记",
            raw_quote="企图篡改覆盖的原文字句",
            kp_id=kp_id,
        )
        assert kp3["version"] == 3
        assert "首次补充的原始引文" in kp3["raw_quote"]
        assert "篡改" not in kp3["raw_quote"]
