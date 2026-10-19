from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str | None = Field(None, description="为空时新建会话")
    message: str = Field(..., min_length=1)
    images: list[str] | None = Field(None, description="可选：图片本地路径、Data URL 或 HTTP URL 列表")
    mode: str | None = Field(
        None,
        description="可选：显式指定学习模式 (explain/quiz/grade/review/plan/homework/chat)，为空时自动路由",
    )
    rag_namespace: str = "default"
    max_steps: int | None = Field(None, ge=1, le=30)
    enable_query_expansion: bool = True
    enable_reflection: bool = True


class RagSearchRequest(BaseModel):
    query: str
    namespace: str = "default"
    top_k: int = Field(5, ge=1, le=50)
    expand: bool = False


class MemoryAddRequest(BaseModel):
    content: str
    type: Literal["working", "episodic", "semantic", "perceptual"] = "episodic"
    importance: float = Field(0.5, ge=0, le=1)
    session_id: str | None = None
    image_path: str | None = None


class ForgetRequest(BaseModel):
    strategy: Literal["importance", "age", "capacity"] = "importance"
    threshold: float | None = None
    max_age_days: int | None = None
    capacity: int | None = None
