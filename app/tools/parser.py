"""工具调用解析器：支持原生 Function Calling、文本标记、AST 容错解析与 Finish 终结判定。"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from app.llm.base import LLMResponse
from app.observability.logging import get_logger
from app.schemas.messages import ToolCall

log = get_logger(__name__)


class ParsedAction(BaseModel):
    thought: str = ""
    is_finish: bool = False
    finish_answer: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


def _safe_eval_dict(raw: str) -> dict[str, Any]:
    """尝试用 JSON 解析，若不规范则使用 Python AST 解析容错。"""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        val = ast.literal_eval(raw)
        if isinstance(val, dict):
            return val
    except Exception:  # noqa: BLE001
        pass

    # 兜底：尝试提取第一个大括号包裹的内容
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            try:
                val = ast.literal_eval(brace_match.group(0))
                if isinstance(val, dict):
                    return val
            except Exception:  # noqa: BLE001
                pass
    return {"_raw": raw}


class ToolCallParser:
    # 匹配 [TOOL_CALL: tool_name: {args}] 或 [TOOL_CALL: tool_name]
    _TEXT_CALL_RE = re.compile(
        r"\[TOOL_CALL:\s*([a-zA-Z0-9_\-]+)(?::\s*(\{.*?\}|.*?))?\]",
        re.DOTALL,
    )
    # 匹配 Action: tool_name(key=val) 或 Action: tool_name: {...} 格式
    _ACTION_FN_RE = re.compile(
        r"Action:\s*([a-zA-Z0-9_\-]+)\s*(?:\((.*?)\)|:\s*(\{.*\}))",
        re.DOTALL,
    )
    # 匹配 Finish[答案] 标记
    _FINISH_RE = re.compile(
        r"Finish\s*\[\s*(.*?)\s*\]",
        re.DOTALL,
    )

    def parse(self, response: LLMResponse) -> ParsedAction:
        text = response.content or ""
        native_reasoning = (response.reasoning_content or "").strip()
        log.debug("parsing_llm_response", raw_len=len(text), native_calls=len(response.tool_calls))

        # 1. 优先使用原生 Function Calling
        if response.tool_calls:
            thought = text.strip() or native_reasoning
            log.info("parsed_native_tool_calls", count=len(response.tool_calls))
            return ParsedAction(thought=thought, is_finish=False, tool_calls=response.tool_calls)

        # 2. 检查 Finish[answer] 标记
        finish_m = self._FINISH_RE.search(text)
        if finish_m:
            answer = finish_m.group(1).strip()
            thought = text[: finish_m.start()].replace("Thought:", "").strip() or native_reasoning
            log.info("parsed_finish_tag", answer_len=len(answer))
            return ParsedAction(thought=thought, is_finish=True, finish_answer=answer)

        # 3. 检查 [TOOL_CALL: name: {...}] 文本标记
        text_matches = list(self._TEXT_CALL_RE.finditer(text))
        if text_matches:
            extracted_calls: list[ToolCall] = []
            thought_parts: list[str] = []
            last_idx = 0

            for m in text_matches:
                thought_parts.append(text[last_idx : m.start()])
                name = m.group(1).strip()
                args_raw = m.group(2) or "{}"
                arguments = _safe_eval_dict(args_raw)
                extracted_calls.append(ToolCall(name=name, arguments=arguments))
                last_idx = m.end()

            thought_parts.append(text[last_idx:])
            thought = "".join(thought_parts).replace("Thought:", "").strip() or native_reasoning
            log.info("parsed_text_tool_calls", count=len(extracted_calls))
            return ParsedAction(thought=thought, is_finish=False, tool_calls=extracted_calls)

        # 4. 检查 Action: name(...) 或 Action: name: {...}
        fn_m = self._ACTION_FN_RE.search(text)
        if fn_m:
            name = fn_m.group(1).strip()
            args_str = (fn_m.group(2) or fn_m.group(3) or "").strip()
            arguments = _safe_eval_dict("{" + args_str + "}" if not args_str.startswith("{") else args_str)
            thought = text[: fn_m.start()].replace("Thought:", "").strip() or native_reasoning
            log.info("parsed_action_fn_call", tool_name=name)
            return ParsedAction(
                thought=thought,
                is_finish=False,
                tool_calls=[ToolCall(name=name, arguments=arguments)],
            )

        # 5. 没有检测到工具调用，整段文本作为最终回答
        thought = native_reasoning
        if "Thought:" in text and ("答案:" in text or "Final Answer:" in text or "Finish:" in text):
            parts = re.split(r"(?:答案:|Final Answer:|Finish:)", text, maxsplit=1)
            thought = parts[0].replace("Thought:", "").strip() or native_reasoning
            answer = parts[1].strip()
        else:
            answer = text.strip()

        log.info("parsed_as_direct_finish", answer_len=len(answer))
        return ParsedAction(thought=thought, is_finish=True, finish_answer=answer)
