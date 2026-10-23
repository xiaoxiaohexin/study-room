"""学习领域专用工具集：知识点管理、智能出题、判分诊断（SM-2 + 掌握度更新）、间隔复习与学情规划。"""

from __future__ import annotations

import datetime
from typing import Any

from app.learning.algorithms import grade_objective
from app.learning.store import LearningStore
from app.llm.base import LLMClient
from app.llm.structured import chat_json
from app.observability.logging import get_logger
from app.rag.retriever import RagRetriever
from app.schemas.messages import Message
from app.tools.base import Tool, ToolContext, ToolResult

log = get_logger(__name__)

_QUIZ_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "stem": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["stem", "options", "answer", "explanation"],
}

_SUBJECTIVE_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "score": {"type": "number"},
        "error_type": {
            "type": "string",
            "enum": ["none", "concept", "calculation", "misread", "incomplete"],
        },
        "feedback": {"type": "string"},
    },
    "required": ["is_correct", "score", "error_type", "feedback"],
}


class KnowledgeTool(Tool):
    name = "knowledge_tool"
    description = "管理与查询课程知识点。支持操作：list（列出知识点）、create（新增/更新知识点）、get（按名称或ID查询知识点详情）。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "get"],
                "description": "操作类型：list / create / get",
            },
            "course_id": {"type": "string", "description": "课程ID或命名空间（默认 default）"},
            "name": {"type": "string", "description": "知识点名称（create/get 时使用）"},
            "summary": {"type": "string", "description": "知识点核心定义或摘要（create 时使用）"},
            "difficulty": {"type": "number", "description": "难度系数 0.1~1.0（默认 0.5）"},
        },
        "required": ["action"],
    }

    def __init__(self, store: LearningStore):
        self.store = store

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "list")
        course_id = (
            arguments.get("course_id") or context.course_id or "default"
        ).strip() or "default"

        if action == "create":
            name = (arguments.get("name") or "").strip()
            if not name:
                return ToolResult(ok=False, error="知识点名称 name 不能为空。")
            summary = arguments.get("summary") or ""
            difficulty = float(arguments.get("difficulty") or 0.5)
            try:
                kp = await self.store.upsert_knowledge_point(
                    course_id=course_id,
                    name=name,
                    summary=summary,
                    difficulty=difficulty,
                    user_id=context.user_id,
                )
            except (PermissionError, ValueError) as e:
                return ToolResult(ok=False, error=str(e))
            return ToolResult(
                ok=True,
                output=f"已保存知识点 [{kp['id']}] {kp['name']} (课程: {kp['course_id']}, 难度: {kp['difficulty']}): {kp['summary']}",
                metadata=kp,
            )

        if action == "get":
            name = (arguments.get("name") or "").strip()
            kp = await self.store.get_knowledge_point(
                name, course_id=course_id, user_id=context.user_id
            )
            if not kp:
                return ToolResult(ok=True, output=f"未找到知识点 '{name}'。")
            return ToolResult(
                ok=True,
                output=f"知识点 [{kp['id']}] {kp['name']} (课程: {kp['course_id']}, 难度: {kp['difficulty']})\n摘要: {kp['summary']}",
                metadata=kp,
            )

        kps = await self.store.list_knowledge_points(
            course_id=None if course_id == "all" else course_id,
            user_id=context.user_id,
        )
        if not kps:
            return ToolResult(ok=True, output=f"课程 [{course_id}] 暂无已录入的知识点。")
        lines = [f"- [{k['id']}] {k['name']} (难度: {k['difficulty']}): {k['summary']}" for k in kps]
        return ToolResult(ok=True, output=f"知识点列表 ({len(kps)} 个):\n" + "\n".join(lines))


class QuizTool(Tool):
    name = "quiz_tool"
    description = "智能出题与练习工具。支持 action='generate'（结合知识库教材按知识点与题型出题）或 action='next'（获取当前待做题目或針對薄弱点出题）。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["generate", "next"],
                "description": "generate（生成新题）或 next（获取下一题）",
            },
            "topic": {"type": "string", "description": "要考察的知识点名称或主题"},
            "question_type": {
                "type": "string",
                "enum": ["choice", "fill", "short_answer"],
                "description": "题型：choice（单选）、fill（填空）、short_answer（简答），默认 choice",
            },
            "difficulty": {"type": "number", "description": "难度 0.1~1.0（默认自动按掌握度匹配）"},
            "namespace": {"type": "string", "description": "知识库命名空间（默认 default）"},
        },
        "required": ["action"],
    }

    def __init__(self, store: LearningStore, retriever: RagRetriever, llm: LLMClient):
        self.store = store
        self.retriever = retriever
        self.llm = llm

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "generate")
        topic = (arguments.get("topic") or "").strip()
        q_type = arguments.get("question_type") or "choice"
        # The session scope is authoritative for both retrieval and course binding.
        namespace = (
            arguments.get("namespace") or context.rag_namespace or "default"
        ).strip() or "default"
        effective_course = context.course_id or namespace

        if action == "next" and not topic:
            pending = await self.store.get_latest_unanswered_question(context.user_id)
            if pending:
                opts_txt = "\n".join(pending["options"]) if pending.get("options") else ""
                return ToolResult(
                    ok=True,
                    output=f"【待作答题目】ID: {pending['id']} | 题型: {pending['type']}\n题干: {pending['stem']}\n{opts_txt}",
                    metadata=pending,
                )
            due = await self.store.get_due_reviews(context.user_id, limit=1)
            if due:
                topic = due[0]["kp_name"]

        if not topic:
            topic = "核心基础概念"

        # 确保知识点存在
        kp = await self.store.get_knowledge_point(
            topic, course_id=effective_course, user_id=context.user_id
        )
        if not kp:
            try:
                kp = await self.store.upsert_knowledge_point(
                    course_id=effective_course,
                    name=topic,
                    summary=f"{topic} 相关概念与应用",
                    user_id=context.user_id,
                )
            except (PermissionError, ValueError) as e:
                return ToolResult(ok=False, error=str(e))

        # 自适应难度：若未指定，取当前掌握度 + 0.1
        difficulty = arguments.get("difficulty")
        if difficulty is None:
            report = await self.store.get_progress_report(context.user_id)
            kp_m = next((m["mastery"] for m in report["knowledge_mastery"] if m["kp_id"] == kp["id"]), 0.4)
            difficulty = round(min(0.9, max(0.2, kp_m + 0.1)), 2)
        else:
            difficulty = float(difficulty)

        # 检索教材原文作为出题依据
        docs = await self.retriever.search([topic], namespace=namespace, top_k=2)
        source_chunk_id = docs[0].id if docs else ""
        ref_text = "\n\n".join(d.content for d in docs) if docs else "（知识库中暂无对应原文，请基于该学科标准通用知识出题）"

        prompt = (
            f"请围绕知识点【{topic}】（难度系数 {difficulty}，题型：{q_type}）命制 1 道高质量练习题。\n"
            f"参考教材资料：\n{ref_text}\n\n"
            "要求：\n"
            "1. 若题型为 choice（单选题），options 必须为包含 4 个选项的字符串数组（如 [\"A. ...\", \"B. ...\", \"C. ...\", \"D. ...\"]），answer 为单个大写字母（如 \"A\"）。\n"
            "2. 若题型为 fill 或 short_answer，options 设为空数组 []。\n"
            "3. 严格输出 JSON 对象，包含 stem, options, answer, explanation 四个字段。"
        )

        try:
            data = await chat_json(
                self.llm,
                [Message(role="user", content=prompt)],
                _QUIZ_GEN_SCHEMA,
                name="quiz_item",
                temperature=0.3,
            )
            stem = str(data.get("stem") or f"请简述 {topic} 的核心原理。")
            options = [str(x) for x in (data.get("options") or [])]
            answer = str(data.get("answer") or "A")
            explanation = str(data.get("explanation") or "")
        except Exception as e:  # noqa: BLE001
            log.warning("quiz_llm_json_fallback", error=str(e))
            if q_type == "choice":
                stem = f"关于【{topic}】，下列说法正确的是哪一项？"
                options = [
                    f"A. {topic} 的核心定义与基本性质成立",
                    "B. 该概念在任何条件下均不成立",
                    "C. 该概念与课程主题完全无关",
                    "D. 以上说法均不正确",
                ]
                answer = "A"
                explanation = f"考察 {topic} 的基础概念辨析。"
            else:
                stem = f"请用自己的话简述【{topic}】的核心定义与应用场景。"
                options = []
                answer = f"{topic}的核心定义与基本性质"
                explanation = f"考察对 {topic} 的理解与归纳。"

        saved = await self.store.add_question(
            kp_id=kp["id"],
            course_id=namespace,
            q_type=q_type,
            stem=stem,
            options=options,
            answer=answer,
            explanation=explanation,
            source_chunk_id=source_chunk_id,
            difficulty=difficulty,
        )
        opts_block = ("\n" + "\n".join(options)) if options else ""
        return ToolResult(
            ok=True,
            output=(
                f"已生成练习题（请展示给学生作答，切勿剧透答案）：\n"
                f"- 题目ID: {saved['id']}\n"
                f"- 知识点: {kp['name']} ({kp['id']})\n"
                f"- 题型: {q_type} | 难度: {difficulty}\n"
                f"- 题干: {stem}{opts_block}"
            ),
            metadata=saved,
        )


class GradeTool(Tool):
    name = "grade_tool"
    description = "对学生的作答进行判分、错因诊断，并自动更新知识点掌握度（Mastery）、SM-2 复习排期与错题本。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "user_answer": {"type": "string", "description": "学生提交的答案内容（如 'A' 或具体解答文本）"},
            "question_id": {
                "type": "string",
                "description": "题目ID（如 q_xxx；若留空则自动匹配当前最新待作答题目）",
            },
            "hints_used": {"type": "integer", "description": "学生作答前使用的提示阶数（默认 0）"},
        },
        "required": ["user_answer"],
    }

    def __init__(self, store: LearningStore, llm: LLMClient):
        self.store = store
        self.llm = llm

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        user_answer = str(arguments.get("user_answer") or "").strip()
        question_id = (arguments.get("question_id") or "").strip()
        hints_used = int(arguments.get("hints_used") or 0)

        if not user_answer:
            return ToolResult(ok=False, error="user_answer 不能为空。")

        q = await self.store.get_question(question_id) if question_id else None
        if not q:
            q = await self.store.get_latest_unanswered_question(context.user_id)
        if not q:
            return ToolResult(ok=False, error="未找到待判分的题目，请先调用 quiz_tool 生成题目或指定有效的 question_id。")

        q_type = q["type"]
        if q_type in ("choice", "fill"):
            is_correct, score = grade_objective(q_type, q["answer"], user_answer)
            error_type = "none" if is_correct else "concept"
            feedback = (
                f"回答正确！标准答案：{q['answer']}。{q.get('explanation', '')}"
                if is_correct
                else f"回答有误。你的答案：{user_answer}，标准答案：{q['answer']}。解析：{q.get('explanation', '')}"
            )
        else:
            # 主观简答题使用 LLM 按评分标准输出 JSON
            grade_prompt = (
                f"题目：{q['stem']}\n"
                f"参考答案与解析：{q['answer']} ({q.get('explanation', '')})\n"
                f"学生作答：{user_answer}\n\n"
                "请公正评分并输出 JSON：is_correct(布尔), score(0.0~1.0), error_type(none/concept/calculation/misread/incomplete), feedback(简明评语)。"
            )
            try:
                g_data = await chat_json(
                    self.llm,
                    [Message(role="user", content=grade_prompt)],
                    _SUBJECTIVE_GRADE_SCHEMA,
                    name="grade_result",
                    temperature=0.1,
                )
                score = float(g_data.get("score", 0.0))
                is_correct = bool(g_data.get("is_correct", score >= 0.6))
                error_type = str(g_data.get("error_type") or ("none" if is_correct else "concept"))
                feedback = str(g_data.get("feedback") or "")
            except Exception:  # noqa: BLE001
                # 兜底：关键词重合度估算
                is_correct = q["answer"].lower() in user_answer.lower() or len(user_answer) >= 10
                score = 0.8 if is_correct else 0.3
                error_type = "none" if is_correct else "incomplete"
                feedback = f"参考答案：{q['answer']}。{q.get('explanation', '')}"

        rec = await self.store.record_attempt_and_update_mastery(
            user_id=context.user_id,
            question_id=q["id"],
            user_answer=user_answer,
            is_correct=is_correct,
            score=score,
            error_type=error_type,
            feedback=feedback,
            hints_used=hints_used,
        )
        next_dt = datetime.datetime.fromtimestamp(rec["next_review_at"]).strftime("%Y-%m-%d %H:%M")
        status_str = "✅ 正确" if is_correct else f"❌ 错误 (已自动加入错题本 {rec['mistake_id']})"
        return ToolResult(
            ok=True,
            output=(
                f"【判分结果】{status_str} | 得分: {score:.2f} | SM-2 质量分: {rec['quality']}\n"
                f"- 题目ID: {q['id']} | 标准答案: {q['answer']}\n"
                f"- 知识点掌握度更新: {rec['prev_mastery']:.2f} -> {rec['new_mastery']:.2f}\n"
                f"- 下次复习间隔: {rec['interval_days']} 天后 ({next_dt})\n"
                f"- 解析与诊断: {feedback}"
            ),
            metadata=rec,
        )


class ReviewTool(Tool):
    name = "review_tool"
    description = (
        "间隔复习与错题本管理工具。支持 action='due'（查看今日到期复习知识点与未解决错题）、"
        "action='resolve'（标记错题已掌握）和 action='grade'（按 SM-2 质量分记录一次闪卡自评）。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["due", "resolve", "grade"],
                "description": "due（查看待复习与错题）、resolve（消除错题）或 grade（记录复习评分）",
            },
            "mistake_id": {"type": "string", "description": "要标记为已掌握的错题ID（resolve 时必填）"},
            "kp_id": {"type": "string", "description": "要评分的知识点ID（grade 时必填）"},
            "quality": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
                "description": "SM-2 质量分 0~5（grade 时必填；<3 视为遗忘，重置复习轮次）",
            },
        },
        "required": ["action"],
    }

    def __init__(self, store: LearningStore):
        self.store = store

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "due")
        if action == "resolve":
            mid = (arguments.get("mistake_id") or "").strip()
            if not mid:
                return ToolResult(ok=False, error="mistake_id 不能为空。")
            ok = await self.store.resolve_mistake(context.user_id, mid)
            return ToolResult(
                ok=True,
                output=f"错题 [{mid}] 已标记为掌握！" if ok else f"未找到错题 [{mid}]。",
            )

        if action == "grade":
            kp_id = (arguments.get("kp_id") or "").strip()
            if not kp_id:
                return ToolResult(ok=False, error="kp_id 不能为空。")
            try:
                quality = int(arguments.get("quality"))
            except (TypeError, ValueError):
                return ToolResult(ok=False, error="quality 必须是 0~5 的整数。")
            if not 0 <= quality <= 5:
                return ToolResult(ok=False, error="quality 必须是 0~5 的整数。")
            try:
                rec = await self.store.review_knowledge_point(kp_id, quality, user_id=context.user_id)
            except KeyError:
                return ToolResult(ok=False, error=f"知识点 [{kp_id}] 不存在。")
            except (PermissionError, ValueError) as e:
                return ToolResult(ok=False, error=str(e))
            next_dt = datetime.datetime.fromtimestamp(rec["next_review_at"]).strftime("%Y-%m-%d %H:%M")
            return ToolResult(
                ok=True,
                output=(
                    f"【SM-2 复习评分已记录】{rec['name']} ({rec['kp_id']})\n"
                    f"- 质量分: {rec['quality']} | 掌握度: {rec['prev_mastery']:.2f} -> {rec['new_mastery']:.2f}\n"
                    f"- 下次复习间隔: {rec['interval_days']} 天后 ({next_dt})"
                ),
                metadata=rec,
            )

        due_list = await self.store.get_due_reviews(context.user_id, limit=10)
        mistakes = await self.store.list_mistakes(context.user_id, resolved=False, limit=10)

        parts: list[str] = []
        if due_list:
            parts.append(f"### 今日到期复习知识点 ({len(due_list)} 个):")
            for d in due_list:
                parts.append(
                    f"- [{d['kp_id']}] {d['kp_name']} | 当前掌握度: {d['mastery']:.2f} | 复习轮次: {d['repetitions']}"
                )
        else:
            parts.append("### 今日到期复习知识点: 暂无到期项")

        if mistakes:
            parts.append(f"\n### 未消灭错题本 ({len(mistakes)} 道):")
            for m in mistakes:
                parts.append(
                    f"- 错题ID [{m['id']}] (知识点: {m.get('kp_name')}, 错因: {m['error_type']}): {m.get('stem') or ''} -> 正确答案: {m.get('expected_answer') or ''}"
                )
        else:
            parts.append("\n### 未消灭错题本: 暂无错题")

        return ToolResult(ok=True, output="\n".join(parts))


class ProgressTool(Tool):
    name = "progress_tool"
    description = "学情诊断与学习计划工具。支持 action='report'（查询掌握度与薄弱点报告）、action='create_plan'（创建学习计划）、action='list_plans'（查看学习计划）。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["report", "create_plan", "list_plans"],
                "description": "report / create_plan / list_plans",
            },
            "goal": {"type": "string", "description": "学习目标（create_plan 时使用）"},
            "deadline": {"type": "string", "description": "目标截止日期（可选）"},
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "分步学习任务列表（create_plan 时使用）",
            },
        },
        "required": ["action"],
    }

    def __init__(self, store: LearningStore):
        self.store = store

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "report")

        if action == "create_plan":
            goal = (arguments.get("goal") or "系统化掌握课程薄弱知识点").strip()
            deadline = (arguments.get("deadline") or "").strip()
            raw_tasks = arguments.get("tasks") or []
            if not raw_tasks:
                rep = await self.store.get_progress_report(context.user_id)
                weak_names = [w["kp_name"] for w in rep["weakest_kps"][:3]] or ["核心概念梳理"]
                raw_tasks = [
                    f"第 1 阶段：精读教材并梳理【{'、'.join(weak_names)}】核心概念",
                    f"第 2 阶段：完成针对性专项练习题并订正错题本（当前未消错题 {rep['unresolved_mistakes_count']} 道）",
                    "第 3 阶段：按 SM-2 间隔复习计划完成到期巩固自测",
                ]
            items = [{"step": idx + 1, "task": str(t), "done": False} for idx, t in enumerate(raw_tasks)]
            plan = await self.store.save_study_plan(
                user_id=context.user_id, goal=goal, items=items, deadline=deadline
            )
            tasks_str = "\n".join(f"  {it['step']}. {it['task']}" for it in items)
            return ToolResult(
                ok=True,
                output=f"已创建学习计划 [{plan['id']}] 目标：{goal}\n{tasks_str}",
                metadata=plan,
            )

        if action == "list_plans":
            plans = await self.store.list_study_plans(context.user_id)
            if not plans:
                return ToolResult(ok=True, output="当前暂无学习计划。")
            lines = []
            for p in plans:
                t_str = "; ".join(f"{it.get('step')}.{it.get('task')}" for it in p["items"])
                lines.append(f"- [{p['id']}] 目标: {p['goal']} (截止: {p['deadline'] or '未设'}) -> {t_str}")
            return ToolResult(ok=True, output="学习计划列表:\n" + "\n".join(lines))

        rep = await self.store.get_progress_report(context.user_id)
        weak_lines = [
            f"  • {w['kp_name']} ({w['kp_id']}): 掌握度 {w['mastery']:.2f} | EF={w['ef']:.2f} | 间隔={w['interval_days']}天"
            for w in rep["weakest_kps"]
        ] or ["  • 暂无知识点作答记录"]
        return ToolResult(
            ok=True,
            output=(
                f"【学情诊断报告 (用户: {context.user_id})】\n"
                f"- 累计作答: {rep['total_attempts']} 题 | 答对: {rep['correct_attempts']} 题 | 正确率: {rep['accuracy'] * 100:.1f}% | 平均分: {rep['avg_score']:.2f}\n"
                f"- 今日待复习知识点: {rep['due_review_count']} 个 | 未解决错题: {rep['unresolved_mistakes_count']} 道\n"
                f"- 薄弱知识点 Top-5:\n" + "\n".join(weak_lines)
            ),
            metadata=rep,
        )
