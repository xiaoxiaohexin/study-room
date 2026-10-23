from __future__ import annotations

from pathlib import Path

from app.context.builder import ContextBuilder
from app.core.config import Settings
from app.learning.algorithms import compute_sm2, grade_objective, score_to_quality, update_mastery_ema
from app.learning.router import IntentRouter
from app.learning.store import LearningStore
from app.main import create_app
from app.storage.sqlite import SQLiteStore
from app.storage.vector import SQLiteVectorStore, VectorPoint


def test_sm2_and_mastery_algorithms():
    # 客观题判卷
    ok, score = grade_objective("choice", "B", "我选 B")
    assert ok is True
    assert score == 1.0

    ok2, score2 = grade_objective("choice", "A", "C")
    assert ok2 is False
    assert score2 == 0.0

    # SM-2 连续答对间隔增长：1 -> 6 -> round(6 * EF)
    q = score_to_quality(True, 1.0, hints_used=0)
    assert q == 5
    s1 = compute_sm2(ef=2.5, interval_days=0.0, repetitions=0, quality=q, now=1000.0)
    assert s1.repetitions == 1
    assert s1.interval_days == 1.0

    s2 = compute_sm2(ef=s1.ef, interval_days=s1.interval_days, repetitions=s1.repetitions, quality=q, now=1000.0)
    assert s2.repetitions == 2
    assert s2.interval_days == 6.0

    s3 = compute_sm2(ef=s2.ef, interval_days=s2.interval_days, repetitions=s2.repetitions, quality=q, now=1000.0)
    assert s3.repetitions == 3
    assert s3.interval_days > 6.0

    # 答错重置间隔为 1 天
    s_fail = compute_sm2(ef=s3.ef, interval_days=s3.interval_days, repetitions=s3.repetitions, quality=1, now=1000.0)
    assert s_fail.repetitions == 0
    assert s_fail.interval_days == 1.0

    # EMA 掌握度更新
    m_up = update_mastery_ema(0.3, 1.0, difficulty=0.7)
    assert m_up > 0.3
    m_down = update_mastery_ema(m_up, 0.0, difficulty=0.7)
    assert m_down < m_up


async def test_sqlite_vector_store_persistence(tmp_path: Path):
    db_file = tmp_path / "vec_persist.db"
    sqlite1 = SQLiteStore(db_file)
    await sqlite1.initialize()
    vstore1 = SQLiteVectorStore(sqlite1)

    await vstore1.upsert(
        "rag_course",
        [
            VectorPoint(id="c1", vector=[1.0, 0.0, 0.0], payload={"namespace": "math", "text": "导数与极限"}),
            VectorPoint(id="c2", vector=[0.0, 1.0, 0.0], payload={"namespace": "physics", "text": "牛顿第二定律"}),
        ],
    )
    await sqlite1.aclose()

    # 模拟进程重启后重新打开 SQLite
    sqlite2 = SQLiteStore(db_file)
    await sqlite2.initialize()
    vstore2 = SQLiteVectorStore(sqlite2)

    hits = await vstore2.search("rag_course", [0.95, 0.05, 0.0], top_k=5, filters={"namespace": "math"})
    assert len(hits) == 1
    assert hits[0].id == "c1"
    assert hits[0].payload["text"] == "导数与极限"
    await sqlite2.aclose()


def test_context_builder_excludes_hyde_and_injects_learner_profile():
    builder = ContextBuilder(max_tokens=4000)
    built = builder.build(
        system_prompt="你是导师。",
        user_query="请讲解链式法则",
        history=[],
        hyde_doc="这是一段假设性锚点文本，不应出现在最终系统提示词中",
        learner_profile="- 当前薄弱知识点: 复合函数求导(掌握度:0.25)",
    )
    sys_text = built.messages[0].content
    assert "HYPOTHETICAL ANCHOR" not in sys_text
    assert "不应出现在最终系统提示词中" not in sys_text
    assert "### [Learner Profile & Mastery]" in sys_text
    assert "复合函数求导" in sys_text


async def test_learning_router_and_store_closed_loop(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "learn.db")
    await sqlite.initialize()
    store = LearningStore(sqlite)
    router = IntentRouter(store)

    # 1. 路由基本意图识别
    d_explain = await router.route("u1", "请给我讲一下什么是反向传播算法？")
    assert d_explain.mode == "explain"

    d_quiz = await router.route("u1", "考考我，出一道关于反向传播的选择题")
    assert d_quiz.mode == "quiz"

    # 2. 创建知识点与题目后，用户直接回复 "A" 自动进入 grade 模式
    kp = await store.upsert_knowledge_point("cs101", "反向传播", "神经网络梯度计算核心算法", difficulty=0.6)
    q = await store.add_question(
        kp_id=kp["id"],
        course_id="cs101",
        user_id="u1",
        q_type="choice",
        stem="反向传播算法的核心数学基础是什么？",
        options=["A. 链式法则", "B. 泰勒展开", "C. 傅里叶变换", "D. 拉格朗日乘子"],
        answer="A",
        explanation="反向传播利用微积分中的链式法则逐层传递梯度。",
    )

    d_grade = await router.route("u1", "B")
    assert d_grade.mode == "grade"
    assert q["id"] in d_grade.reason

    # 3. 判错 -> 自动进错题本 & 更新掌握度
    rec_wrong = await store.record_attempt_and_update_mastery(
        user_id="u1",
        question_id=q["id"],
        user_answer="B",
        is_correct=False,
        score=0.0,
        error_type="concept",
        feedback="混淆了链式法则与泰勒展开",
    )
    assert rec_wrong["mistake_id"] is not None
    assert rec_wrong["new_mastery"] < rec_wrong["prev_mastery"]

    mistakes = await store.list_mistakes("u1", resolved=False)
    assert len(mistakes) == 1
    assert mistakes[0]["id"] == rec_wrong["mistake_id"]

    # 4. 消除错题
    resolved_ok = await store.resolve_mistake("u1", rec_wrong["mistake_id"])
    assert resolved_ok is True
    assert len(await store.list_mistakes("u1", resolved=False)) == 0

    await sqlite.aclose()


async def test_intent_router_retrieval_policy():
    router = IntentRouter()

    smalltalk = await router.route("u1", "你好")
    assert smalltalk.mode == "chat"
    assert smalltalk.retrieval_mode == "none"
    assert smalltalk.allow_web_fallback is False
    assert smalltalk.expose_tools is False
    assert smalltalk.tool_names == []

    acknowledgement = await router.route("u1", "好的，谢谢")
    assert acknowledgement.mode == "chat"
    assert acknowledgement.retrieval_mode == "none"
    assert acknowledgement.expose_tools is False

    explanation = await router.route("u1", "请给我讲一下什么是反向传播算法？")
    assert explanation.mode == "explain"
    assert explanation.retrieval_mode == "local"
    assert explanation.allow_web_fallback is False
    assert explanation.expose_tools is True

    web_lookup = await router.route("u1", "帮我查一下最新的光合作用研究进展")
    assert web_lookup.mode == "chat"
    assert web_lookup.retrieval_mode == "local"
    assert web_lookup.allow_web_fallback is True

    review = await router.route("u1", "帮我复习今天的错题")
    assert review.mode == "review"
    assert review.retrieval_mode == "none"
    assert review.expose_tools is True


async def test_learning_api_endpoints(tmp_path: Path):
    from httpx import ASGITransport, AsyncClient

    from app.core.container import AppContainer

    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "learn_api.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. 创建课程与知识点
            r_course = await client.post(
                "/api/courses",
                json={"user_id": "stu_1", "name": "高等数学", "description": "微积分基础"},
            )
            assert r_course.status_code == 200
            course_id = r_course.json()["course"]["id"]

            r_kp = await client.post(
                f"/api/courses/{course_id}/knowledge",
                headers={"X-User-Id": "stu_1"},
                json={"name": "洛必达法则", "summary": "0/0 与 ∞/∞ 未定式求极限", "difficulty": 0.6},
            )
            assert r_kp.status_code == 200

            # 2. 生成测验题（确认返回给前端的数据不泄露标准答案）
            r_quiz = await client.post(
                "/api/quiz/generate",
                json={"user_id": "stu_1", "topic": "洛必达法则", "question_type": "choice"},
            )
            assert r_quiz.status_code == 200
            q_data = r_quiz.json()["question"]
            qid = q_data["id"]
            assert "answer" not in q_data

            # 3. 提交作答并判卷
            r_ans = await client.post(
                f"/api/quiz/{qid}/answer",
                json={"user_id": "stu_1", "user_answer": "B"},
            )
            assert r_ans.status_code == 200
            ans_res = r_ans.json()["result"]
            assert ans_res["question_id"] == qid
            assert "new_mastery" in ans_res

            # 4. 查看学情报告与待复习列表
            r_prog = await client.get("/api/progress/stu_1")
            assert r_prog.status_code == 200
            prog = r_prog.json()
            assert prog["total_attempts"] == 1

            r_due = await client.get("/api/review/stu_1/due")
            assert r_due.status_code == 200
            assert "unresolved_mistakes" in r_due.json()

            # 5. 生成学习计划
            r_plan = await client.post(
                "/api/plans/stu_1",
                json={"goal": "一周攻克求极限与洛必达法则"},
            )
            assert r_plan.status_code == 200
            assert r_plan.json()["plan"]["goal"] == "一周攻克求极限与洛必达法则"
    finally:
        await container.aclose()


async def test_llama_compat_learning_reconnection(tmp_path: Path):
    import json

    from httpx import ASGITransport, AsyncClient

    from app.core.container import AppContainer

    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "compat_learn.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. 测试知识讲解意图识别、会话持久化与 JSONL 对话日志
            resp1 = await client.post(
                "/v1/chat/completions",
                json={
                    "user_id": "test_student",
                    "session_id": "sess_compat_1",
                    "messages": [{"role": "user", "content": "请给我讲解一下洛必达法则的原理"}],
                    "stream": False,
                    "tools": [
                        {"type": "function", "function": {"name": "rag_search_tool"}},
                        {"type": "function", "function": {"name": "quiz_tool"}},
                        {"type": "function", "function": {"name": "terminal_tool"}},
                    ],
                },
            )
            assert resp1.status_code == 200
            assert resp1.headers.get("x-learning-mode") == "explain"

            # 验证 SessionStateManager 记录了用户和助手消息
            sess = await container.state_manager.get_or_create_session("test_student", "sess_compat_1")
            assert len(sess.history) == 2
            assert sess.history[0].role == "user"
            assert sess.history[1].role == "assistant"

            # 验证 DialogueLogger 写入了当天的 JSONL 日志文件
            log_file = container.dialogue_logger._file()
            assert log_file.exists()
            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            last_record = json.loads(lines[-1])
            assert last_record["user_id"] == "test_student"
            assert "洛必达法则" in last_record["query"]
            assert len(last_record["steps"]) == 1
            assert last_record["steps"][0]["finish"] is True

            # 2. 测试多模态图片上传自动路由至 homework 模式与感知记忆写入
            resp2 = await client.post(
                "/v1/chat/completions",
                json={
                    "user_id": "test_student",
                    "session_id": "sess_compat_1",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "帮我看看这道题怎么做"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}},
                            ],
                        }
                    ],
                    "stream": False,
                },
            )
            assert resp2.status_code == 200
            assert resp2.headers.get("x-learning-mode") == "homework"

            # 验证感知记忆存在
            perceptual_mems = await container.memory_manager.perceptual.retrieve("帮我看看这道题怎么做", user_id="test_student")
            assert len(perceptual_mems) >= 1

            # 3. 测试工具调用 POST /tools 会自动存入工作记忆 (Working Memory)
            resp3 = await client.post(
                "/tools",
                json={
                    "tool": "knowledge_tool",
                    "params": {"action": "list", "user_id": "test_student", "session_id": "sess_compat_1"},
                },
            )
            assert resp3.status_code == 200
            working_mems = await container.memory_manager.working.retrieve("knowledge_tool", user_id="test_student")
            assert len(working_mems) >= 1
    finally:
        await container.aclose()


async def test_llama_compat_skips_retrieval_for_smalltalk(tmp_path: Path):
    from unittest.mock import AsyncMock

    from httpx import ASGITransport, AsyncClient

    from app.core.container import AppContainer

    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "compat_smalltalk.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    container.rag_retriever.search = AsyncMock(return_value=[])  # type: ignore[method-assign]
    container.memory_manager.retrieve_all = AsyncMock(return_value=[])  # type: ignore[method-assign]

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "user_id": "test_student",
                    "session_id": "sess_smalltalk",
                    "messages": [{"role": "user", "content": "你好"}],
                    "stream": False,
                    "tools": [
                        {"type": "function", "function": {"name": "rag_search_tool"}},
                        {"type": "function", "function": {"name": "web_tool"}},
                    ],
                },
            )

        assert response.status_code == 200
        assert response.headers.get("x-retrieval-mode") == "none"
        container.rag_retriever.search.assert_not_awaited()
        container.memory_manager.retrieve_all.assert_not_awaited()
    finally:
        await container.aclose()


async def test_knowledge_point_provenance_and_version_history(tmp_path: Path):
    """验证知识点不可变原文摘录、可编辑正文分层存储、版本号递增、历史快照与掌握度自动入库。"""
    sqlite = SQLiteStore(tmp_path / "kp_version.db")
    await sqlite.initialize()
    store = LearningStore(sqlite)

    # 1. 划词初次创建 (v1)
    prov = {
        "message_id": "msg_001",
        "turn_index": 2,
        "doc_title": "deep_learning.pdf",
        "page_start": 42,
        "page_end": 43,
        "quote_snippet": "梯度下降算法通过反向传播误差更新权重。",
    }
    kp1 = await store.upsert_knowledge_point(
        course_id="cs229",
        name="梯度下降",
        summary="参数优化核心方法",
        difficulty=0.5,
        raw_quote=prov["quote_snippet"],
        content="基础版：沿负梯度方向更新。",
        provenance=prov,
        user_id="user_test_m2",
    )
    assert kp1["version"] == 1
    assert kp1["raw_quote"] == prov["quote_snippet"]
    assert kp1["content"] == "基础版：沿负梯度方向更新。"
    assert kp1["provenance"]["page_start"] == 42
    assert len(kp1["history"]) == 1
    assert kp1["history"][0]["version"] == 1

    # 验证 mastery 表已自动登记 (SM-2 初始值: ef=2.5, repetitions=0, interval=1.0, mastery=0.3)
    cur_m = await sqlite.conn.execute(
        "SELECT * FROM mastery WHERE user_id = ? AND kp_id = ?",
        ("user_test_m2", kp1["id"]),
    )
    m_row = await cur_m.fetchone()
    assert m_row is not None
    assert m_row["mastery"] == 0.3
    assert m_row["ef"] == 2.5
    assert m_row["interval_days"] == 1.0
    assert m_row["repetitions"] == 0

    # 2. 编辑正文 (v2) - 验证不可变摘录保持不变，版本号递增
    kp2 = await store.upsert_knowledge_point(
        course_id="cs229",
        name="梯度下降",
        summary="补充动量项说明",
        difficulty=0.6,
        raw_quote="尝试修改原始摘录（应当被忽略）",
        content="进阶版：引入动量（Momentum）和自适应学习率（Adam）。",
        kp_id=kp1["id"],
        user_id="user_test_m2",
    )
    assert kp2["id"] == kp1["id"]
    assert kp2["version"] == 2
    # 原始摘录不变
    assert kp2["raw_quote"] == prov["quote_snippet"]
    assert kp2["content"] == "进阶版：引入动量（Momentum）和自适应学习率（Adam）。"
    assert len(kp2["history"]) == 2
    assert kp2["history"][1]["version"] == 2

    # 3. 验证 get_knowledge_point 和 get_revisions
    fetched = await store.get_knowledge_point(kp1["id"], user_id="user_test_m2")
    assert fetched is not None
    assert fetched["version"] == 2
    assert fetched["provenance"]["doc_title"] == "deep_learning.pdf"

    revisions = await store.get_revisions(kp1["id"], user_id="user_test_m2")
    assert len(revisions) == 2
    assert revisions[0]["version"] == 1
    assert revisions[1]["version"] == 2

    await sqlite.aclose()


async def test_knowledge_point_non_destructive_rollback(tmp_path: Path):
    """验证知识点版本非破坏性回滚（生成新版本快照并恢复目标内容）。"""
    import pytest

    sqlite = SQLiteStore(tmp_path / "kp_rollback.db")
    await sqlite.initialize()
    store = LearningStore(sqlite)

    # 创建 v1
    kp = await store.upsert_knowledge_point(
        course_id="math101",
        name="泰勒展开",
        summary="初版",
        raw_quote="f(x) = f(a) + f'(a)(x-a) + ...",
        content="第一版笔记：多项式近似。",
    )
    kp_id = kp["id"]
    assert kp["version"] == 1

    # 更新到 v2
    await store.upsert_knowledge_point(
        course_id="math101",
        name="泰勒展开",
        content="第二版笔记：包含皮亚诺余项。",
        kp_id=kp_id,
    )

    # 更新到 v3
    await store.upsert_knowledge_point(
        course_id="math101",
        name="泰勒展开",
        content="第三版笔记：包含拉格朗日余项推导。",
        kp_id=kp_id,
    )

    revs_before = await store.get_revisions(kp_id)
    assert len(revs_before) == 3

    # 执行非破坏性回滚至 v1 -> 生成 v4，正文恢复为 v1
    rolled = await store.rollback_knowledge_point(kp_id, target_version=1)
    assert rolled["version"] == 4
    assert rolled["content"] == "第一版笔记：多项式近似。"
    assert len(rolled["history"]) == 4
    assert rolled["history"][-1]["changeType"] == "rollback"

    # 再次查询确认持久化
    fetched = await store.get_knowledge_point(kp_id)
    assert fetched["version"] == 4
    assert fetched["content"] == "第一版笔记：多项式近似。"

    # 边界情况：回滚到不存在的版本抛出 ValueError
    with pytest.raises(ValueError):
        await store.rollback_knowledge_point(kp_id, target_version=99)

    # 边界情况：回滚不存在的知识点抛出 KeyError
    with pytest.raises(KeyError):
        await store.rollback_knowledge_point("kp_nonexistent", target_version=1)

    await sqlite.aclose()


async def test_semantic_relations_and_prereq_dual_write(tmp_path: Path):
    """验证强类型语义关系存储、查询、删除及 prerequisite 双写 kp_prereq。"""
    sqlite = SQLiteStore(tmp_path / "relations.db")
    await sqlite.initialize()
    store = LearningStore(sqlite)

    # 创建 3 个知识点
    kp_limit = await store.upsert_knowledge_point("c1", "极限")
    kp_diff = await store.upsert_knowledge_point("c1", "微分")
    kp_integ = await store.upsert_knowledge_point("c1", "积分")

    # 1. 插入 prerequisite 关系：极限 是 微分 的前置依赖
    rel1 = await store.upsert_relation(
        src_id=kp_limit["id"],
        rel_type="prerequisite",
        dst_id=kp_diff["id"],
        metadata={"weight": 1.0},
    )
    assert rel1["rel_type"] == "prerequisite"

    # 验证 dual-write: kp_prereq 存在记录 (kp_id=微分, prereq_id=极限)
    cur_p = await sqlite.conn.execute("SELECT * FROM kp_prereq WHERE kp_id = ?", (kp_diff["id"],))
    prereqs = await cur_p.fetchall()
    assert len(prereqs) == 1
    assert prereqs[0]["prereq_id"] == kp_limit["id"]

    # 2. 插入 causal 与 similar 关系
    await store.upsert_relation(
        src_id=kp_diff["id"],
        rel_type="causal",
        dst_id=kp_integ["id"],
        metadata={"note": "微积分基本定理连结微分与积分"},
    )
    await store.upsert_relation(
        src_id=kp_diff["id"],
        rel_type="similar",
        dst_id=kp_integ["id"],
    )

    # 3. 查询关联边
    out_edges = await store.get_relations(kp_id=kp_diff["id"], direction="out")
    assert len(out_edges) == 2  # causal + similar
    in_edges = await store.get_relations(kp_id=kp_diff["id"], direction="in")
    assert len(in_edges) == 1   # prerequisite

    # 4. 删除 prerequisite 关系 -> 同时自动清理 kp_prereq
    deleted = await store.delete_relation(
        src_id_or_id=kp_limit["id"],
        rel_type="prerequisite",
        dst_id=kp_diff["id"],
    )
    assert deleted is True
    cur_p2 = await sqlite.conn.execute("SELECT * FROM kp_prereq WHERE kp_id = ?", (kp_diff["id"],))
    assert len(await cur_p2.fetchall()) == 0

    await sqlite.aclose()


async def test_learning_api_relations_rollback_and_revisions(tmp_path: Path):
    """验证 REST API: 关系建立/查询/删除、版本历史时间轴及回滚接口闭环。"""
    from httpx import ASGITransport, AsyncClient

    from app.core.container import AppContainer

    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "learn_api_m2.db",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. 创建两个知识点（包含出处元数据与原始引文）
            r_kp1 = await client.post(
                "/api/courses/cs_algo/knowledge",
                json={
                    "name": "二分查找",
                    "summary": "有序数组对分检索",
                    "raw_quote": "每次将搜索区间减半，时间复杂度 O(log n)。",
                    "content": "前提：数组必须单调有序。",
                    "provenance": {"doc_title": "algo.pdf", "page_start": 15},
                },
            )
            assert r_kp1.status_code == 200
            kp1_id = r_kp1.json()["id"]

            r_kp2 = await client.post(
                "/api/courses/cs_algo/knowledge",
                json={
                    "name": "二叉搜索树",
                    "summary": "左小右大动态数据结构",
                    "content": "支持动态增删查。",
                    "prereq_ids": [kp1_id],
                },
            )
            assert r_kp2.status_code == 200
            kp2_id = r_kp2.json()["id"]

            # 2. 建立因果语义连线 (POST /api/courses/{course_id}/knowledge/relations)
            r_rel = await client.post(
                "/api/courses/cs_algo/knowledge/relations",
                json={
                    "src_id": kp1_id,
                    "rel_type": "causal",
                    "dst_id": kp2_id,
                    "metadata": {"reason": "二分思想推广到树结构"},
                },
            )
            assert r_rel.status_code == 200
            assert r_rel.json()["created"] is True

            # 3. 查询单节点关联边与全课关联边
            r_edges = await client.get(f"/api/courses/cs_algo/knowledge/{kp1_id}/relations")
            assert r_edges.status_code == 200
            assert r_edges.json()["count"] >= 1

            r_all_edges = await client.get("/api/courses/cs_algo/knowledge/relations")
            assert r_all_edges.status_code == 200
            assert r_all_edges.json()["count"] >= 1

            # 4. 更新知识点生成 v2，并获取历史版本列表
            r_update = await client.post(
                "/api/courses/cs_algo/knowledge",
                json={
                    "id": kp1_id,
                    "name": "二分查找",
                    "content": "注意边界条件：mid = left + (right - left) // 2 防止溢出。",
                },
            )
            assert r_update.status_code == 200
            assert r_update.json()["version"] == 2

            r_revs = await client.get(f"/api/courses/cs_algo/knowledge/{kp1_id}/revisions")
            assert r_revs.status_code == 200
            assert r_revs.json()["count"] == 2

            # 5. 回滚到 v1 (POST /api/courses/{course_id}/knowledge/{kp_id}/rollback)
            r_rollback = await client.post(
                f"/api/courses/cs_algo/knowledge/{kp1_id}/rollback",
                json={"target_version": 1},
            )
            assert r_rollback.status_code == 200
            assert r_rollback.json()["version"] == 3
            assert r_rollback.json()["knowledge_point"]["content"] == "前提：数组必须单调有序。"

            # 6. 删除语义连线
            r_del = await client.delete(
                "/api/courses/cs_algo/knowledge/relations",
                params={"src_id": kp1_id, "rel_type": "causal", "dst_id": kp2_id},
            )
            assert r_del.status_code == 200
            assert r_del.json()["deleted"] is True
    finally:
        await container.aclose()


