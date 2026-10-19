"""学习意图路由器（IntentRouter）：分模式提示词与精简工具子集路由。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.learning.store import LearningStore
from app.observability.logging import get_logger

log = get_logger(__name__)

LearningMode = Literal["explain", "quiz", "grade", "review", "plan", "homework", "chat"]


@dataclass
class RouteDecision:
    mode: LearningMode
    system_prompt: str
    tool_names: list[str]
    max_steps: int = 6
    reason: str = ""


# 每个模式下仅暴露 2~4 个最相关的工具，防止本地小模型（Gemma-4 E4B）工具选择过载
MODE_TOOLS: dict[LearningMode, list[str]] = {
    "explain": ["rag_search_tool", "knowledge_tool", "note_tool"],
    "quiz": ["quiz_tool", "knowledge_tool", "rag_search_tool"],
    "grade": ["grade_tool", "review_tool"],
    "review": ["review_tool", "quiz_tool", "grade_tool"],
    "plan": ["progress_tool", "knowledge_tool", "note_tool"],
    "homework": ["rag_search_tool", "knowledge_tool", "note_tool"],
    "chat": [
        "rag_search_tool",
        "knowledge_tool",
        "quiz_tool",
        "grade_tool",
        "review_tool",
        "progress_tool",
        "note_tool",
        "memory_tool",
    ],
}

MODE_PROMPTS: dict[LearningMode, str] = {
    "explain": AGENT_SYSTEM_PROMPT
    + """

【当前模式：知识讲解模式 (Explain)】
1. 严格依据上下文中的 `[DOCUMENT]` 教材资料与 `[Learner Profile & Mastery]` 水平进行讲解；若教材未覆盖该内容，请明确说明。
2. 讲解结构清晰：先给直观核心定义，再给推导/原理解析与典型例题，最后抛出 1 个启发性小问题检验理解。
3. 如需补充查阅教材或知识点结构，请调用 `rag_search_tool` 或 `knowledge_tool`。""",
    "quiz": AGENT_SYSTEM_PROMPT
    + """

【当前模式：练习出题模式 (Practice / Quiz)】
1. 请优先调用 `quiz_tool`（action="generate" 或 action="next"）为学生生成或抽取针对性练习题。
2. 出题展示时：**只展示题干、题号（question_id）与选项**，绝对不要直接把标准答案和解析剧透给学生，引导学生在下一轮作答。""",
    "grade": AGENT_SYSTEM_PROMPT
    + """

【当前模式：作答判分与诊断模式 (Grade)】
1. 学生正在提交题目答案。请调用 `grade_tool` 对学生的回答进行判分、更新掌握度（Mastery）与 SM-2 复习排期。
2. 结合 `grade_tool` 返回的判分结果、标准答案与错因（概念混淆/计算失误/审题错误/知识遗忘），给学生清晰的讲解与鼓励。""",
    "review": AGENT_SYSTEM_PROMPT
    + """

【当前模式：间隔复习与错题攻坚模式 (Review)】
1. 请调用 `review_tool` 查看学生今日到期的 SM-2 复习知识点与未解决错题。
2. 针对薄弱点帮学生梳理核心易错概念，或调用 `quiz_tool` 出变式题巩固。""",
    "plan": AGENT_SYSTEM_PROMPT
    + """

【当前模式：学情诊断与学习规划模式 (Progress & Plan)】
1. 请调用 `progress_tool` 获取学生的掌握度报告、薄弱知识点 Top-N 或创建结构化学习计划。
2. 基于真实作答统计和 SM-2 到期情况给出可执行的每日/每周学习建议。""",
    "homework": AGENT_SYSTEM_PROMPT
    + """

【当前模式：多模态作业苏格拉底辅导模式 (Socratic Homework Tutor)】
1. 学生上传了题目/作业图片或正在求助解题。请遵循**苏格拉底启发式教学**：
   - 第 1 阶：指出题目考察的核心知识点与解题方向；
   - 第 2 阶：给出关键公式或切入点，留出关键步骤让学生自己思考；
   - 除非学生明确要求“直接给完整解析”，否则不要一次性直接端出最终答案。""",
    "chat": AGENT_SYSTEM_PROMPT,
}


class IntentRouter:
    """高响应速度的学习意图路由器：确定性规则 + 会话待答题状态感知。"""

    _QUIZ_RE = re.compile(r"(出.{0,4}题|考考我|做.{0,3}题|练习题|来.{0,3}道题|测验|quiz|刷题)", re.IGNORECASE)
    _REVIEW_RE = re.compile(r"(复习|错题|待复习|review|今天学什么|到期)", re.IGNORECASE)
    _PLAN_RE = re.compile(
        r"(学习计划|规划|进度|掌握度|薄弱点|学情|诊断报告|考前冲刺|plan|progress)", re.IGNORECASE
    )
    _GRADE_RE = re.compile(
        r"^(?:我?选\s*[A-Fa-f]|答案(?:是|为)?\s*\S+|我的(?:答案|回答)(?:是|为)|提交答案|判分|批改)",
        re.IGNORECASE,
    )
    _EXPLAIN_RE = re.compile(
        r"(讲一下|讲解|什么是|为什么|怎么理解|原理是|推导|解释一下|教我|概念|知识点)", re.IGNORECASE
    )

    def __init__(self, learning_store: LearningStore | None = None):
        self.learning_store = learning_store

    async def route(
        self,
        user_id: str,
        message: str,
        images: list[str] | None = None,
        explicit_mode: str | None = None,
    ) -> RouteDecision:
        msg = message.strip()

        # 0. 显式指定模式优先
        if explicit_mode and explicit_mode in MODE_TOOLS:
            mode: LearningMode = explicit_mode  # type: ignore[assignment]
            return RouteDecision(
                mode=mode,
                system_prompt=MODE_PROMPTS[mode],
                tool_names=MODE_TOOLS[mode],
                reason="explicit_mode",
            )

        # 1. 多模态图片上传默认进入苏格拉底作业辅导模式
        if images:
            return RouteDecision(
                mode="homework",
                system_prompt=MODE_PROMPTS["homework"],
                tool_names=MODE_TOOLS["homework"],
                reason="multimodal_image_attached",
            )

        # 2. 检查是否存在待作答题目，且用户输入像是在作答
        if self.learning_store:
            pending_q = await self.learning_store.get_latest_unanswered_question(user_id)
            if pending_q:
                is_short_choice = bool(re.match(r"^[A-Fa-f]$", msg))
                if is_short_choice or self._GRADE_RE.search(msg) or f"q_" in msg:
                    return RouteDecision(
                        mode="grade",
                        system_prompt=MODE_PROMPTS["grade"],
                        tool_names=MODE_TOOLS["grade"],
                        max_steps=4,
                        reason=f"answering_pending_question:{pending_q['id']}",
                    )

        # 3. 关键词规则匹配
        if self._GRADE_RE.search(msg):
            return RouteDecision(
                mode="grade",
                system_prompt=MODE_PROMPTS["grade"],
                tool_names=MODE_TOOLS["grade"],
                max_steps=4,
                reason="keyword_grade",
            )
        if self._QUIZ_RE.search(msg):
            return RouteDecision(
                mode="quiz",
                system_prompt=MODE_PROMPTS["quiz"],
                tool_names=MODE_TOOLS["quiz"],
                max_steps=5,
                reason="keyword_quiz",
            )
        if self._REVIEW_RE.search(msg):
            return RouteDecision(
                mode="review",
                system_prompt=MODE_PROMPTS["review"],
                tool_names=MODE_TOOLS["review"],
                max_steps=5,
                reason="keyword_review",
            )
        if self._PLAN_RE.search(msg):
            return RouteDecision(
                mode="plan",
                system_prompt=MODE_PROMPTS["plan"],
                tool_names=MODE_TOOLS["plan"],
                max_steps=5,
                reason="keyword_plan",
            )
        if self._EXPLAIN_RE.search(msg):
            return RouteDecision(
                mode="explain",
                system_prompt=MODE_PROMPTS["explain"],
                tool_names=MODE_TOOLS["explain"],
                max_steps=6,
                reason="keyword_explain",
            )

        # 4. 默认通用学习对话模式
        return RouteDecision(
            mode="chat",
            system_prompt=MODE_PROMPTS["chat"],
            tool_names=MODE_TOOLS["chat"],
            max_steps=8,
            reason="default_chat",
        )
