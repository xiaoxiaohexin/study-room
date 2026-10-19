"""脚本化 FakeLLM：用于单元测试与无 API Key 的离线开发。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from app.llm.base import LLMChunk, LLMResponse
from app.schemas.messages import Message, ToolSpec

Responder = Callable[[list[Message], list[ToolSpec] | None], LLMResponse]


class FakeLLM:
    """按顺序返回预置响应；队列耗尽后使用 default 响应。

    responses 中元素可以是 LLMResponse、str 或 Responder 函数。
    """

    def __init__(self, responses: list | None = None, default: str = "Finish[这是 FakeLLM 的默认回答]"):
        self.responses = list(responses or [])
        self.default = default
        self.calls: list[list[Message]] = []

    def _next(self, messages, tools) -> LLMResponse:
        self.calls.append(messages)
        if self.responses:
            r = self.responses.pop(0)
        else:
            r = self.default
        if callable(r):
            r = r(messages, tools)
        if isinstance(r, str):
            r = LLMResponse(content=r)
        return r

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        return self._next(messages, tools)

    async def stream(self, messages, tools=None, **kwargs) -> AsyncIterator[LLMChunk]:
        r = self._next(messages, tools)
        for i in range(0, len(r.content), 8):
            yield LLMChunk(delta=r.content[i : i + 8])
        yield LLMChunk(finish_reason="stop")

    async def chat_json(self, messages, schema, **kwargs):
        from app.llm.structured import extract_json

        return extract_json(self._next(messages, None).content)

    async def aclose(self) -> None:
        return None
