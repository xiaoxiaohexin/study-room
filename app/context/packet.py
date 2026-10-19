"""上下文包模型（ContextPacket）与分区定义。"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

PacketSource = Literal["system", "history", "memory", "rag", "note", "tool", "hyde", "profile"]


class ContextPacket(BaseModel):
    id: str
    source: PacketSource
    content: str
    tokens: int
    relevance: float = 0.5
    recency: float = 0.5
    importance: float = 0.5
    priority: int = 10  # 优先级，数字越小优先级越高（如 system=0, task=1）
    created_at: float = Field(default_factory=time.time)

    @property
    def score(self) -> float:
        # 打分公式：(0.7 * 相关性 + 0.3 * 新鲜度) * (0.8 + 0.4 * importance)
        return (0.7 * self.relevance + 0.3 * self.recency) * (0.8 + 0.4 * self.importance)
