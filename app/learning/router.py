"""闭环控制意图路由器（IntentRouter）：分模式提示词与精简工具子集动态挂载。

核心闭环链路：
基于 RAG 检索知识源生成测试 (quiz) -> 自动化混合裁决评估 (grade) -> 基于记忆衰减（SM-2）计算下一次重放时间并软收敛失败任务 (review) -> 状态遥测与编排优化流水线 (plan)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.learning.store import LearningStore
from app.observability.logging import get_logger
from app.query.policy import RetrievalMode, has_web_intent, is_non_retrieval_chat

log = get_logger(__name__)

LearningMode = Literal["explain", "quiz", "grade", "review", "plan", "homework", "chat"]


@dataclass
class RouteDecision:
    mode: LearningMode
    system_prompt: str
    tool_names: list[str]
    max_steps: int = 6
    reason: str = ""
    retrieval_mode: RetrievalMode = "local"
    allow_web_fallback: bool = False
    expose_tools: bool = True


# 每个模式下仅暴露 2~4 个最相关的工具，防止本地小参数模型工具选择过载
MODE_TOOLS: dict[LearningMode, list[str]] = {
    "explain": ["rag_search_tool", "knowledge_tool", "note_tool", "web_tool"],
    "quiz": ["quiz_tool", "knowledge_tool", "rag_search_tool"],
    "grade": ["grade_tool", "review_tool"],
    "review": ["review_tool", "quiz_tool", "grade_tool"],
    "plan": ["progress_tool", "knowledge_tool", "note_tool"],
    "homework": ["rag_search_tool", "knowledge_tool", "note_tool", "web_tool"],
    "chat": [
        "rag_search_tool",
        "knowledge_tool",
        "quiz_tool",
        "grade_tool",
        "review_tool",
        "progress_tool",
        "note_tool",
        "memory_tool",
        "web_tool",
    ],
}

MODE_PROMPTS: dict[LearningMode, str] = {
    "explain": AGENT_SYSTEM_PROMPT
    + """

【当前模式：知识溯源与深度解析模式 (Explain)】
1. 严格依据上下文中的 `[DOCUMENT]` 知识源切片与系统状态遥测信息进行解析；若知识库未覆盖该内容，请明确说明。
2. 结构化输出核心定义、机制原理与关键应用示例，并标注文档页码出处。
3. 如需补充检索文档切片或登记知识节点，请调用 `rag_search_tool` 或 `knowledge_tool`。""",
    "quiz": AGENT_SYSTEM_PROMPT
    + """

【当前模式：基准验证测试生成模式 (Test Generation / Quiz)】
1. 请优先调用 `quiz_tool`（action="generate" 或 action="next"）结合知识源生成或抽取结构化验证测试任务。
2. 展示测试任务时：**仅展示任务题干、任务编号（question_id）与候选项**，请勿提前暴露基准答案与判定解析，等待下一轮响应提交。""",
    "grade": AGENT_SYSTEM_PROMPT
    + """

【当前模式：自动化混合裁决与诊断模式 (Hybrid Adjudication / Grade)】
1. 用户正在提交待测任务的响应结果。请调用 `grade_tool` 执行混合裁决评估、更新节点稳健度（EMA）并基于记忆衰减（SM-2）计算下一次重放时间。
2. 结合 `grade_tool` 返回的裁决得分、基准答案与偏差分类（concept/calculation/misread/incomplete），输出清晰的诊断结论。""",
    "review": AGENT_SYSTEM_PROMPT
    + """

【当前模式：衰减重放调度与失败任务收敛模式 (Decay Replay / Review)】
1. 请调用 `review_tool` 查询当前到期需重放的知识节点（SM-2）与未收敛失败任务集，或对已修复项执行软状态收敛标记（action="resolve"）。
2. 针对低稳健度节点梳理核心机制，或调用 `quiz_tool` 生成回归验证测试。""",
    "plan": AGENT_SYSTEM_PROMPT
    + """

【当前模式：闭环状态遥测与优化流水线编排模式 (Telemetry & Pipeline Plan)】
1. 请调用 `progress_tool` 获取节点稳健度遥测报告、低稳健度节点 Top-N 或编排结构化优化执行流水线。
2. 基于真实测试统计、失败任务分布与 SM-2 到期重放情况给出可执行的分阶段优化方案。""",
    "homework": AGENT_SYSTEM_PROMPT
    + """

【当前模式：多模态视觉解析与分步推演模式 (Multimodal Visual Analysis)】
1. 请求包含图像输入。请结合视觉内容与知识源上下文进行结构化信息提取与分步逻辑推演：
   - 第 1 阶段：识别图像中的关键实体、文本、图表结构与核心问题；
   - 第 2 阶段：给出关键分析切入点、约束条件与分步推导路径。""",
    "chat": AGENT_SYSTEM_PROMPT,
}


class IntentRouter:
    """低延迟闭环控制意图路由器：确定性规则匹配 + 会话待评估任务状态感知。"""

    _QUIZ_RE = re.compile(
        r"(生成.{0,4}测试|基准测试|生成.{0,4}验证|出.{0,4}题|考考我|做.{0,3}题|练习题|来.{0,3}道题|测验|quiz|刷题)",
        re.IGNORECASE,
    )
    _REVIEW_RE = re.compile(
        r"(重放|失败任务|异常案例|待重放|软收敛|复习|错题|待复习|review|今天学什么|到期)",
        re.IGNORECASE,
    )
    _PLAN_RE = re.compile(
        r"(优化流水线|流水线编排|遥测报告|稳健度|薄弱节点|学习计划|规划|进度|掌握度|薄弱点|学情|诊断报告|考前冲刺|plan|progress)",
        re.IGNORECASE,
    )
    _GRADE_RE = re.compile(
        r"^(?:我?选\s*[A-Fa-f]|答案(?:是|为)?\s*\S+|我的(?:答案|回答|结果)(?:是|为)|提交(?:答案|响应|结果)|裁决|评估结果|判分|批改)",
        re.IGNORECASE,
    )
    _EXPLAIN_RE = re.compile(
        r"(解析一下|讲一下|讲解|什么是|为什么|怎么理解|原理是|推导|解释一下|教我|概念|知识点|知识节点)",
        re.IGNORECASE,
    )

    def __init__(self, learning_store: LearningStore | None = None):
        self.learning_store = learning_store

    @staticmethod
    def _make_decision(
        mode: LearningMode,
        reason: str,
        message: str,
        *,
        max_steps: int = 6,
    ) -> RouteDecision:
        """Build a decision with deterministic retrieval and tool boundaries."""
        tool_names = list(MODE_TOOLS[mode])
        retrieval_mode: RetrievalMode = "local"
        expose_tools = True

        if mode == "chat" and is_non_retrieval_chat(message):
            retrieval_mode = "none"
            tool_names = []
            expose_tools = False
        elif mode in {"grade", "review", "plan"}:
            # These modes obtain state through their dedicated closed-loop tools.
            retrieval_mode = "none"

        allow_web = mode not in {"grade", "review", "plan"} and has_web_intent(message)
        return RouteDecision(
            mode=mode,
            system_prompt=MODE_PROMPTS[mode],
            tool_names=tool_names,
            max_steps=max_steps,
            reason=reason,
            retrieval_mode=retrieval_mode,
            allow_web_fallback=allow_web,
            expose_tools=expose_tools,
        )

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
            return self._make_decision(
                mode,
                "explicit_mode",
                msg,
            )

        # 1. 多模态图片输入默认进入多模态视觉解析模式
        if images:
            return self._make_decision(
                "homework",
                "multimodal_image_attached",
                msg,
            )

        # 2. 检查是否存在待评估测试任务，且用户输入符合响应提交特征
        if self.learning_store:
            pending_q = await self.learning_store.get_latest_unanswered_question(user_id)
            if pending_q:
                is_short_choice = bool(re.match(r"^[A-Fa-f]$", msg))
                if is_short_choice or self._GRADE_RE.search(msg) or "q_" in msg:
                    return self._make_decision(
                        "grade",
                        f"answering_pending_question:{pending_q['id']}",
                        msg,
                        max_steps=4,
                    )

        # 3. 关键词规则匹配
        if self._GRADE_RE.search(msg):
            return self._make_decision(
                "grade",
                "keyword_grade",
                msg,
                max_steps=4,
            )
        if self._QUIZ_RE.search(msg):
            return self._make_decision(
                "quiz",
                "keyword_quiz",
                msg,
                max_steps=5,
            )
        if self._REVIEW_RE.search(msg):
            return self._make_decision(
                "review",
                "keyword_review",
                msg,
                max_steps=5,
            )
        if self._PLAN_RE.search(msg):
            return self._make_decision(
                "plan",
                "keyword_plan",
                msg,
                max_steps=5,
            )
        if self._EXPLAIN_RE.search(msg):
            return self._make_decision(
                "explain",
                "keyword_explain",
                msg,
                max_steps=6,
            )

        # 4. 默认通用智能体交互模式
        return self._make_decision(
            "chat",
            "default_chat",
            msg,
            max_steps=8,
        )
