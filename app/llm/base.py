from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.schemas.messages import Message, ToolCall, ToolSpec


class LLMResponse(BaseModel):
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None


class LLMChunk(BaseModel):
    delta: str = ""
    reasoning_delta: str = ""
    finish_reason: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    async def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None, **kwargs
    ) -> LLMResponse: ...

    def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None, **kwargs
    ) -> AsyncIterator[LLMChunk]: ...

    async def aclose(self) -> None: ...
