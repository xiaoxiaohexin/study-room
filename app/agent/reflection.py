"""反思评估模块（Reflection）：评估当前执行轨迹的信息完备度与终止建议。"""

from __future__ import annotations

import json

from pydantic import BaseModel

from app.agent.prompts import REFLECTION_PROMPT
from app.llm.base import LLMClient
from app.observability.logging import get_logger
from app.schemas.messages import Message

log = get_logger(__name__)


class ReflectionVerdict(BaseModel):
    done: bool = False
    critique: str = ""
    next_hint: str = ""


class Reflector:
    def __init__(self, llm: LLMClient, interval_steps: int = 3):
        self.llm = llm
        # 两次自省之间的最小步数间隔（避免同一异常连续触发多次 LLM 调用）
        self.interval_steps = interval_steps

    def should_reflect(
        self,
        step: int,
        *,
        tool_errors: int = 0,
        retrieval_empty: bool = False,
        repeated_calls: int = 0,
    ) -> bool:
        """只在出现异常信号时自省：工具报错、检索为空或重复调用同一工具。

        正常的多步轨迹不再按固定步长无条件反思（审计 P1：异常驱动的 Reflection）。
        """
        if step <= 0:
            return False
        return tool_errors > 0 or retrieval_empty or repeated_calls > 0

    async def reflect(self, user_query: str, scratchpad_text: str) -> ReflectionVerdict:
        log.info("reflection_started", user_query=user_query[:50])
        prompt = REFLECTION_PROMPT.format(user_query=user_query, scratchpad=scratchpad_text)
        try:
            resp = await self.llm.chat([Message(role="user", content=prompt)], temperature=0.2)
            cleaned = resp.content.strip().strip("`").replace("json\n", "").strip()
            data = json.loads(cleaned)
            verdict = ReflectionVerdict(
                done=bool(data.get("done", False)),
                critique=str(data.get("critique", "")),
                next_hint=str(data.get("next_hint", "")),
            )
            log.info("reflection_completed", done=verdict.done, critique=verdict.critique)
            return verdict
        except Exception as e:  # noqa: BLE001
            log.warning("reflection_evaluation_failed", error=str(e))
            return ReflectionVerdict(done=False, critique=f"Evaluation failed: {e}", next_hint="")
