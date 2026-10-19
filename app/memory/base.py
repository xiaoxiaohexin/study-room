"""记忆系统基础定义与抽象基类。"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

MemoryType = Literal["working", "episodic", "semantic", "perceptual"]


class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    user_id: str
    session_id: str | None = None
    type: MemoryType
    content: str
    importance: float = Field(0.5, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredMemory(BaseModel):
    item: MemoryItem
    score: float
    breakdown: dict[str, float] = Field(default_factory=dict)


class BaseMemory(ABC):
    @abstractmethod
    async def add(self, item: MemoryItem) -> str:
        """存储一条记忆，返回 memory id。"""
        ...

    @abstractmethod
    async def retrieve(self, query: str, user_id: str, top_k: int = 5) -> list[ScoredMemory]:
        """检索相关记忆。"""
        ...

    @abstractmethod
    async def remove(self, ids: list[str]) -> None:
        """删除记忆。"""
        ...

    @abstractmethod
    async def list_items(self, user_id: str, limit: int = 50) -> list[MemoryItem]:
        """获取记忆列表。"""
        ...
