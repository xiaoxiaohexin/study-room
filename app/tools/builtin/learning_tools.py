"""闭环控制与知识验证工具集：结构化知识节点管理、基于 RAG 知识源生成测试、自动化混合裁决评估（SM-2 衰减调度 + EMA 稳健度更新）、衰减重放与失败任务软收敛、状态遥测与优化流水线编排。

核心架构与功能咬合说明：
- 最原始的逻辑：一个**“基于 RAG 检索知识源生成测试 -> 自动化混合裁决评估 -> 基于记忆衰减（SM-2）计算下一次重放时间 -> 自动收集失败任务并编排优化流水线”**的闭环控制系统。
- 删数据的逻辑：Agent 端严禁硬删除，业务端采用软状态标记（Soft Resolve），仅在底层维护受用户作用域隔离的级联物理删除。
"""

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
    """结构化知识节点管理工具。

    遵循数据安全策略：Agent 端仅暴露查询与幂等写入（list / create / get），严禁暴露硬删除操作；
    物理级联删除仅在底层受用户作用域隔离的存储管理接口中维护。
    """

    name = "knowledge_tool"
    description = "管理与检索结构化知识节点。支持操作：list（列出知识节点）、create（新增/更新知识节点）、get（按名称或ID查询知识节点详情）。Agent 端严禁硬删除。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "get"],
                "description": "操作类型：list / create / get",
            },
            "course_id": {"type": "string", "description": "所属知识域ID或命名空间（默认 default）"},
            "name": {"type": "string", "description": "知识节点名称（create/get 时使用）"},
            "summary": {"type": "string", "description": "知识节点核心定义或摘要（create 时使用）"},
            "difficulty": {"type": "number", "description": "复杂度权重系数 0.1~1.0（默认 0.5）"},
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
                return ToolResult(ok=False, error="知识节点名称 name 不能为空。")
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
                output=f"已保存知识节点 [{kp['id']}] {kp['name']} (知识域: {kp['course_id']}, 复杂度: {kp['difficulty']}): {kp['summary']}",
                metadata=kp,
            )

        if action == "get":
            name = (arguments.get("name") or "").strip()
            kp = await self.store.get_knowledge_point(
                name, course_id=course_id, user_id=context.user_id
            )
            if not kp:
                return ToolResult(ok=True, output=f"未找到知识节点 '{name}'。")
            return ToolResult(
                ok=True,
                output=f"知识节点 [{kp['id']}] {kp['name']} (知识域: {kp['course_id']}, 复杂度: {kp['difficulty']})\n摘要: {kp['summary']}",
                metadata=kp,
            )

        kps = await self.store.list_knowledge_points(
            course_id=None if course_id == "all" else course_id,
            user_id=context.user_id,
        )
        if not kps:
            return ToolResult(ok=True, output=f"知识域 [{course_id}] 暂无已录入的知识节点。")
        lines = [f"- [{k['id']}] {k['name']} (复杂度: {k['difficulty']}): {k['summary']}" for k in kps]
        return ToolResult(ok=True, output=f"知识节点列表 ({len(kps)} 个):\n" + "\n".join(lines))


class QuizTool(Tool):
    """基于 RAG 检索知识源生成基准测试任务与待测项抽取工具（闭环控制第 1 环）。"""

    name = "quiz_tool"
    description = "基于 RAG 检索知识源生成验证测试任务。支持 action='generate'（结合向量知识库文档按目标节点与任务类型生成结构化测试项）或 action='next'（获取当前待处理测试项或针对到期/低稳健度节点生成测试）。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["generate", "next"],
                "description": "generate（生成新测试项）或 next（获取下一待测任务）",
            },
            "topic": {"type": "string", "description": "要验证的知识节点名称或目标主题"},
            "question_type": {
                "type": "string",
                "enum": ["choice", "fill", "short_answer"],
                "description": "测试任务类型：choice（客观单选）、fill（精确匹配填空）、short_answer（开放式简答），默认 choice",
            },
            "difficulty": {"type": "number", "description": "复杂度系数 0.1~1.0（默认自动按当前节点稳健度自适应计算）"},
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
        # 会话级作用域强制收口：检索命名空间与知识域绑定以会话上下文为准，防止越权跨库检索。
        namespace = (context.rag_namespace or "default").strip() or "default"
        effective_course = context.course_id or namespace

        if action == "next" and not topic:
            pending = await self.store.get_latest_unanswered_question(context.user_id)
            if pending:
                opts_txt = "\n".join(pending["options"]) if pending.get("options") else ""
                return ToolResult(
                    ok=True,
                    output=f"【待评估测试项】ID: {pending['id']} | 类型: {pending['type']}\n任务描述: {pending['stem']}\n{opts_txt}",
                    metadata=pending,
                )
            due = await self.store.get_due_reviews(context.user_id, limit=1)
            if due:
                topic = due[0]["kp_name"]

        if not topic:
            topic = "核心基础概念"

        # 确保关联知识节点存在
        kp = await self.store.get_knowledge_point(
            topic, course_id=effective_course, user_id=context.user_id
        )
        if not kp:
            try:
                kp = await self.store.upsert_knowledge_point(
                    course_id=effective_course,
                    name=topic,
                    summary=f"{topic} 相关机制与应用",
                    user_id=context.user_id,
                )
            except (PermissionError, ValueError) as e:
                return ToolResult(ok=False, error=str(e))

        # 自适应复杂度：若未显式指定，取当前节点稳健度指标 + 0.1
        difficulty = arguments.get("difficulty")
        if difficulty is None:
            report = await self.store.get_progress_report(context.user_id)
            kp_m = next((m["mastery"] for m in report["knowledge_mastery"] if m["kp_id"] == kp["id"]), 0.4)
            difficulty = round(min(0.9, max(0.2, kp_m + 0.1)), 2)
        else:
            difficulty = float(difficulty)

        # 基于 RAG 检索知识源文档切片作为生成基准测试的事实依据
        docs = await self.retriever.search([topic], namespace=namespace, top_k=2)
        source_chunk_id = docs[0].id if docs else ""
        ref_text = "\n\n".join(d.content for d in docs) if docs else "（当前知识源中暂无对应文档切片，请基于领域标准通用事实生成测试）"

        prompt = (
            f"请围绕目标知识节点【{topic}】（复杂度系数 {difficulty}，测试类型：{q_type}）构建 1 项高质量基准验证测试任务。\n"
            f"参考知识源片段：\n{ref_text}\n\n"
            "要求：\n"
            "1. 若类型为 choice（单选），options 必须为包含 4 个候选项的字符串数组（如 [\"A. ...\", \"B. ...\", \"C. ...\", \"D. ...\"]），answer 为单个大写字母（如 \"A\"）。\n"
            "2. 若类型为 fill 或 short_answer，options 设为空数组 []。\n"
            "3. 严格输出 JSON 对象，包含 stem（任务题干）, options（候选项）, answer（标准基准答案）, explanation（判定依据与解析）四个字段。"
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
                stem = f"关于【{topic}】，下列命题成立的是哪一项？"
                options = [
                    f"A. {topic} 的核心定义与基本性质成立",
                    "B. 该机制在任何条件下均不成立",
                    "C. 该节点与当前知识域完全无关",
                    "D. 以上命题均不正确",
                ]
                answer = "A"
                explanation = f"验证 {topic} 的核心概念与约束条件。"
            else:
                stem = f"请简述【{topic}】的核心定义与典型应用场景。"
                options = []
                answer = f"{topic}的核心定义与基本性质"
                explanation = f"验证对 {topic} 的结构化归纳。"

        saved = await self.store.add_question(
            kp_id=kp["id"],
            course_id=namespace,
            user_id=context.user_id,
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
                f"已生成验证测试任务（仅展示任务内容以供提交响应，请勿提前暴露基准答案）：\n"
                f"- 任务ID: {saved['id']}\n"
                f"- 关联节点: {kp['name']} ({kp['id']})\n"
                f"- 任务类型: {q_type} | 复杂度: {difficulty}\n"
                f"- 任务内容: {stem}{opts_block}"
            ),
            metadata=saved,
        )


class GradeTool(Tool):
    """自动化混合裁决评估工具（闭环控制第 2 & 3 环：客观规则 + 主观 LLM 混合裁决 -> EMA 稳健度更新 & SM-2 衰减重放计算 -> 失败任务自动归档）。"""

    name = "grade_tool"
    description = "对提交的测试响应执行自动化混合裁决评估、偏差归因分析，并自动更新知识节点稳健度（EMA）、计算基于记忆衰减（SM-2）的下一次重放时间及归档失败任务。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "user_answer": {"type": "string", "description": "提交的候选响应内容（如 'A' 或具体解答文本）"},
            "question_id": {
                "type": "string",
                "description": "测试任务ID（如 q_xxx；若留空则自动匹配当前最新待评估测试任务）",
            },
            "hints_used": {"type": "integer", "description": "前置辅助引导阶数（默认 0，用于折算 SM-2 质量分）"},
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

        q = (
            await self.store.get_question(question_id, user_id=context.user_id)
            if question_id
            else None
        )
        if question_id and not q:
            # 显式指定了 question_id 但未命中：任务不存在或不属于当前用户作用域，严禁跨任务回退。
            return ToolResult(ok=False, error=f"测试任务 [{question_id}] 不存在或不属于当前用户。")
        if not q:
            q = await self.store.get_latest_unanswered_question(context.user_id)
        if not q:
            return ToolResult(ok=False, error="未找到待评估的测试任务，请先调用 quiz_tool 生成测试任务或指定有效的 question_id。")

        q_type = q["type"]
        if q_type in ("choice", "fill"):
            is_correct, score = grade_objective(q_type, q["answer"], user_answer)
            error_type = "none" if is_correct else "concept"
            feedback = (
                f"裁决通过！基准答案：{q['answer']}。{q.get('explanation', '')}"
                if is_correct
                else f"裁决未通过。提交内容：{user_answer}，基准答案：{q['answer']}。诊断依据：{q.get('explanation', '')}"
            )
        else:
            # 开放式任务使用 LLM 按结构化 Schema 进行混合裁决
            grade_prompt = (
                f"测试任务：{q['stem']}\n"
                f"基准答案与判定依据：{q['answer']} ({q.get('explanation', '')})\n"
                f"提交响应：{user_answer}\n\n"
                "请客观执行裁决评估并输出 JSON：is_correct(布尔), score(0.0~1.0), error_type(none/concept/calculation/misread/incomplete), feedback(简明诊断结论)。"
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
            except Exception as e:  # noqa: BLE001
                # 兜底保护：LLM 结构化裁决服务不可用时严禁臆断通过，否则会污染 SM-2 衰减周期与稳健度指标（审计 P0）。
                # 仅返回待复核状态（ungraded），不写入执行记录、不触发状态更新。
                log.warning(
                    "subjective_grade_ungraded",
                    question_id=q["id"],
                    user_id=context.user_id,
                    error=str(e),
                )
                return ToolResult(
                    ok=True,
                    output=(
                        "【裁决评估结果】⚠️ 待复核 (ungraded) | 自动裁决服务不可用，本项未计入统计。\n"
                        f"- 任务ID: {q['id']} | 基准要点: {q['answer']}\n"
                        f"- 提交内容: {user_answer}\n"
                        "- 本次响应未写入失败任务集，也未更新节点稳健度与 SM-2 重放排期。\n"
                        f"- 判定依据: {q.get('explanation', '') or '（无）'}"
                    ),
                    metadata={
                        "status": "ungraded",
                        "question_id": q["id"],
                        "user_id": context.user_id,
                        "score": None,
                    },
                )

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
        status_str = "✅ 评估通过" if is_correct else f"❌ 评估未通过 (已自动收集至失败任务集 {rec['mistake_id']})"
        return ToolResult(
            ok=True,
            output=(
                f"【裁决评估结果】{status_str} | 得分: {score:.2f} | SM-2 质量分: {rec['quality']}\n"
                f"- 任务ID: {q['id']} | 基准答案: {q['answer']}\n"
                f"- 节点稳健度更新 (EMA): {rec['prev_mastery']:.2f} -> {rec['new_mastery']:.2f}\n"
                f"- 下次重放时间 (SM-2): {rec['interval_days']} 天后 ({next_dt})\n"
                f"- 诊断结论: {feedback}"
            ),
            metadata=rec,
        )


class ReviewTool(Tool):
    """基于记忆衰减（SM-2）的重放调度与失败任务软收敛工具。

    遵循数据删除策略：业务端对失败任务集采用软状态标记（Soft Resolve: resolved=1），Agent 端严禁硬删除。
    """

    name = "review_tool"
    description = (
        "基于衰减模型（SM-2）的重放调度与失败任务集管理工具。支持 action='due'（查询当前到期需重放的知识节点与未收敛失败任务）、"
        "action='resolve'（软状态标记失败任务为已收敛 Soft Resolve，严禁硬删除）和 action='grade'（按 SM-2 质量分推进节点重放周期）。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["due", "resolve", "grade"],
                "description": "due（查询到期重放项与失败任务）、resolve（软标记失败任务已收敛）或 grade（记录 SM-2 重放评分）",
            },
            "mistake_id": {"type": "string", "description": "要软标记为已收敛的失败任务记录ID（resolve 时必填）"},
            "kp_id": {"type": "string", "description": "要更新重放周期的知识节点ID（grade 时必填）"},
            "quality": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
                "description": "SM-2 质量分 0~5（grade 时必填；<3 视为衰减失效，重置重放轮次）",
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
                output=f"失败任务记录 [{mid}] 已执行软状态收敛标记（Soft Resolve）！" if ok else f"未找到失败任务记录 [{mid}]。",
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
                rec = await self.store.review_knowledge_point(
                    kp_id, quality, user_id=context.user_id
                )
            except KeyError:
                return ToolResult(ok=False, error=f"知识节点 [{kp_id}] 不存在。")
            except (PermissionError, ValueError) as e:
                return ToolResult(ok=False, error=str(e))
            next_dt = datetime.datetime.fromtimestamp(rec["next_review_at"]).strftime("%Y-%m-%d %H:%M")
            return ToolResult(
                ok=True,
                output=(
                    f"【SM-2 衰减重放评分已记录】{rec['name']} ({rec['kp_id']})\n"
                    f"- 质量分: {rec['quality']}/5 | 节点稳健度: {rec['prev_mastery']:.2f} -> {rec['new_mastery']:.2f}\n"
                    f"- 下次重放间隔: {rec['interval_days']} 天后 ({next_dt}) | EF: {rec['ef']}"
                ),
                metadata=rec,
            )

        due_list = await self.store.get_due_reviews(context.user_id, limit=10)
        mistakes = await self.store.list_mistakes(context.user_id, resolved=False, limit=10)

        parts: list[str] = []
        if due_list:
            parts.append(f"### 当前到期需重放知识节点 ({len(due_list)} 个):")
            for d in due_list:
                parts.append(
                    f"- [{d['kp_id']}] {d['kp_name']} | 当前稳健度: {d['mastery']:.2f} | 连续重放轮次: {d['repetitions']}"
                )
        else:
            parts.append("### 当前到期需重放知识节点: 暂无到期项")

        if mistakes:
            parts.append(f"\n### 未收敛失败任务集 ({len(mistakes)} 项):")
            for m in mistakes:
                parts.append(
                    f"- 失败记录ID [{m['id']}] (节点: {m.get('kp_name')}, 偏差类型: {m['error_type']}): {m.get('stem') or ''} -> 基准答案: {m.get('expected_answer') or ''}"
                )
        else:
            parts.append("\n### 未收敛失败任务集: 暂无未收敛项")

        return ToolResult(ok=True, output="\n".join(parts))


class ProgressTool(Tool):
    """闭环状态遥测报告与优化流水线编排工具（闭环控制第 4 环）。"""

    name = "progress_tool"
    description = "闭环状态遥测与优化流水线编排工具。支持 action='report'（查询节点稳健度与低分薄弱项遥测报告）、action='create_plan'（基于失败任务与薄弱节点编排优化执行流水线）、action='list_plans'（查看已编排的优化流水线列表）。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["report", "create_plan", "list_plans"],
                "description": "report / create_plan / list_plans",
            },
            "goal": {"type": "string", "description": "流水线优化目标（create_plan 时使用）"},
            "deadline": {"type": "string", "description": "目标截止时间或里程碑（可选）"},
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "分步优化执行任务列表（create_plan 时使用；留空则自动根据失败任务集与低稳健度节点生成）",
            },
        },
        "required": ["action"],
    }

    def __init__(self, store: LearningStore):
        self.store = store

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        action = arguments.get("action", "report")

        if action == "create_plan":
            goal = (arguments.get("goal") or "系统化收敛低稳健度知识节点与失败任务").strip()
            deadline = (arguments.get("deadline") or "").strip()
            raw_tasks = arguments.get("tasks") or []
            if not raw_tasks:
                rep = await self.store.get_progress_report(context.user_id)
                weak_names = [w["kp_name"] for w in rep["weakest_kps"][:3]] or ["核心基础节点"]
                raw_tasks = [
                    f"阶段 1：基于 RAG 知识源核验并补齐【{'、'.join(weak_names)}】结构化节点定义",
                    f"阶段 2：执行针对性基准验证测试并收敛未解决失败任务（当前待收敛 {rep['unresolved_mistakes_count']} 项）",
                    "阶段 3：按 SM-2 衰减周期完成到期节点的回归重放与稳健度复核",
                ]
            items = [{"step": idx + 1, "task": str(t), "done": False} for idx, t in enumerate(raw_tasks)]
            plan = await self.store.save_study_plan(
                user_id=context.user_id, goal=goal, items=items, deadline=deadline
            )
            tasks_str = "\n".join(f"  {it['step']}. {it['task']}" for it in items)
            return ToolResult(
                ok=True,
                output=f"已编排优化执行流水线 [{plan['id']}] 目标：{goal}\n{tasks_str}",
                metadata=plan,
            )

        if action == "list_plans":
            plans = await self.store.list_study_plans(context.user_id)
            if not plans:
                return ToolResult(ok=True, output="当前暂无已编排的优化流水线。")
            lines = []
            for p in plans:
                t_str = "; ".join(f"{it.get('step')}.{it.get('task')}" for it in p["items"])
                lines.append(f"- [{p['id']}] 目标: {p['goal']} (里程碑: {p['deadline'] or '未设'}) -> {t_str}")
            return ToolResult(ok=True, output="优化执行流水线列表:\n" + "\n".join(lines))

        rep = await self.store.get_progress_report(context.user_id)
        weak_lines = [
            f"  • {w['kp_name']} ({w['kp_id']}): 稳健度 {w['mastery']:.2f} | EF={w['ef']:.2f} | 重放间隔={w['interval_days']}天"
            for w in rep["weakest_kps"]
        ] or ["  • 暂无节点评估记录"]
        return ToolResult(
            ok=True,
            output=(
                f"【闭环状态遥测报告 (作用域用户: {context.user_id})】\n"
                f"- 累计执行测试: {rep['total_attempts']} 次 | 裁决通过: {rep['correct_attempts']} 次 | 通过率: {rep['accuracy'] * 100:.1f}% | 平均得分: {rep['avg_score']:.2f}\n"
                f"- 到期需重放节点: {rep['due_review_count']} 个 | 未收敛失败任务: {rep['unresolved_mistakes_count']} 项\n"
                f"- 低稳健度薄弱节点 Top-5:\n" + "\n".join(weak_lines)
            ),
            metadata=rep,
        )
