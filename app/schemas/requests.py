from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.messages import validate_image_input


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str | None = Field(None, description="为空时新建会话")
    message: str = Field(..., min_length=1)
    images: list[str] | None = Field(None, description="可选：受控上传目录内的图片路径、Data URL 或 HTTP URL 列表")
    mode: str | None = Field(
        None,
        description="可选：显式指定执行模式 (explain/quiz/grade/review/plan/homework/chat)，为空时自动路由",
    )
    course_id: str | None = Field(None, description="可选：关联知识域 ID")
    rag_namespace: str | None = Field(
        None,
        description="可选：关联 RAG 知识库命名空间，为空时遵循会话绑定或回退 default",
    )
    max_steps: int | None = Field(None, ge=1, le=30)
    enable_query_expansion: bool = True
    enable_reflection: bool = True

    @field_validator("images")
    @classmethod
    def _validate_image_paths(cls, value: list[str] | None) -> list[str] | None:
        """本地图片路径只能是受控上传目录内的文件（审计 P1-4）。"""
        for item in value or []:
            try:
                validate_image_input(item)
            except (PermissionError, FileNotFoundError, OSError, ValueError) as e:
                raise ValueError(f"Invalid image reference: {e}") from e
        return value


class RagSearchRequest(BaseModel):
    query: str
    namespace: str = "default"
    top_k: int = Field(5, ge=1, le=50)
    expand: bool = False


class RagAttributeRequest(BaseModel):
    """Selected quote attribution: match a quote back to the source document chunk it came from."""

    quote: str = Field(..., min_length=2, max_length=8000)
    namespace: str = "default"
    top_k: int = Field(8, ge=1, le=30)
    min_overlap: float | None = Field(
        None, ge=0.0, le=1.0, description="字符三元组包含度阈值；为空时使用服务端配置"
    )


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
