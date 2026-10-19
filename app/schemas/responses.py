from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepRecord(BaseModel):
    step: int
    thought: str = ""
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    observation: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    steps: list[StepRecord] = Field(default_factory=list)
    finished: bool = True
    usage: dict[str, int] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    code: int
    error: str
    detail: Any = None
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]


class IngestResponse(BaseModel):
    namespace: str
    filename: str
    chunks: int
    doc_id: str
