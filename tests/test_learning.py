from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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
    from httpx import ASGITransport, AsyncClient
    from datetime import datetime
    import json
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

