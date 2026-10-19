"""OpenAI 兼容 LLM 客户端（DeepSeek / Qwen / OpenAI / vLLM / Ollama 等）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI

from app.core.config import Settings
from app.llm.base import LLMChunk, LLMResponse
from app.observability.logging import get_logger
from app.schemas.messages import Message, ToolCall, ToolSpec

log = get_logger(__name__)


class OpenAICompatLLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model
        is_local = "127.0.0.1" in settings.llm_base_url or "localhost" in settings.llm_base_url
        self._http_client = httpx.AsyncClient(trust_env=not is_local, timeout=settings.llm_timeout)
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value() or "EMPTY",
            timeout=settings.llm_timeout,
            http_client=self._http_client,
        )

    async def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None, **kwargs
    ) -> LLMResponse:
        extra_body = {"cache_prompt": True, **(kwargs.pop("extra_body", None) or {})}
        params = {
            "model": kwargs.pop("model", self.model),
            "messages": [m.to_openai() for m in messages],
            "temperature": kwargs.pop("temperature", self.settings.llm_temperature),
            "extra_body": extra_body,
            **kwargs,
        }
        if tools:
            params["tools"] = [t.to_openai() for t in tools]
            params["tool_choice"] = "auto"
        resp = await self.client.chat.completions.create(**params)
        choice = resp.choices[0]
        calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
            }
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        return LLMResponse(
            content=choice.message.content or "",
            reasoning_content=reasoning,
            tool_calls=calls,
            usage=usage,
            finish_reason=choice.finish_reason,
        )

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None, **kwargs
    ) -> AsyncIterator[LLMChunk]:
        extra_body = {"cache_prompt": True, **(kwargs.pop("extra_body", None) or {})}
        params = {
            "model": kwargs.pop("model", self.model),
            "messages": [m.to_openai() for m in messages],
            "temperature": kwargs.pop("temperature", self.settings.llm_temperature),
            "stream": True,
            "extra_body": extra_body,
            **kwargs,
        }
        if tools:
            params["tools"] = [t.to_openai() for t in tools]
            params["tool_choice"] = "auto"
        stream = await self.client.chat.completions.create(**params)
        async for chunk in stream:
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            reasoning_delta = getattr(ch.delta, "reasoning_content", None) or ""
            yield LLMChunk(
                delta=ch.delta.content or "",
                reasoning_delta=reasoning_delta,
                finish_reason=ch.finish_reason,
            )

    async def chat_json(
        self,
        messages: list[Message],
        schema: dict,
        *,
        name: str = "result",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """JSON Schema 约束输出，并关闭思考（思考会占满 max_tokens 导致 content 为空）。"""
        from app.llm.structured import extract_json

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return extract_json(resp.choices[0].message.content or "")

    async def aclose(self) -> None:
        await self.client.close()
