"""SSE 事件协议 —— 前端对接以此为准。

每个 SSE 帧:  event: <type>\\n data: <AgentEvent JSON>\\n\\n
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    SESSION = "session"  # data: {session_id}
    STATUS = "status"  # data: {stage, detail}  检索/扩展等阶段提示
    THOUGHT = "thought"  # data: {text}
    TOOL_CALL = "tool_call"  # data: {id, name, arguments}
    OBSERVATION = "observation"  # data: {id, name, ok, output}
    REFLECTION = "reflection"  # data: {done, critique, next_hint}
    TOKEN = "token"  # data: {text}  最终答案增量
    FINAL = "final"  # data: {answer, finished}
    ERROR = "error"  # data: {message}
    DONE = "done"  # data: {usage}


class AgentEvent(BaseModel):
    type: EventType
    step: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)

    def to_sse(self) -> dict[str, str]:
        return {"event": self.type.value, "data": self.model_dump_json()}
