"""结构化输出工具：JSON Schema 约束生成 + 宽松解析兜底。

实测（llama-server b9010 + Gemma-4 E4B）：
- `response_format={"type": "json_schema", ...}` 且关闭思考时，约 0.7s 返回合法 JSON；
- 不加约束时模型常输出 ```json 代码块或结构不符的数组，不可靠。
因此所有需要结构化结果的调用（意图分类、出题、主观题判分）都走这里。
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.llm.base import LLMClient
from app.observability.logging import get_logger
from app.schemas.messages import Message

log = get_logger(__name__)


def extract_json(text: str) -> Any:
    """从模型输出中尽力提取 JSON（去掉代码块围栏，截取首个对象/数组）。失败抛 ValueError。"""
    s = (text or "").strip()
    if not s:
        raise ValueError("empty response")
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON found in: {s[:120]!r}")
    start = min(starts)
    end = max(s.rfind("}"), s.rfind("]"))
    if end <= start:
        raise ValueError(f"unbalanced JSON in: {s[:120]!r}")
    return json.loads(s[start : end + 1])


async def chat_json(
    llm: LLMClient,
    messages: list[Message],
    schema: dict[str, Any],
    *,
    name: str = "result",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Any:
    """请求符合 schema 的 JSON。客户端支持原生约束时用原生，否则退化为普通对话 + 宽松解析。"""
    native = getattr(llm, "chat_json", None)
    if callable(native):
        return await native(messages, schema, name=name, temperature=temperature, max_tokens=max_tokens)
    resp = await llm.chat(messages, temperature=temperature)
    return extract_json(resp.content)
