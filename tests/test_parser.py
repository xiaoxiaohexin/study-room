import pytest
from app.llm.base import LLMResponse
from app.schemas.messages import ToolCall
from app.tools.parser import ToolCallParser


def test_parser_native_tool_call():
    parser = ToolCallParser()
    resp = LLMResponse(
        content="I will check notes",
        tool_calls=[ToolCall(name="note_tool", arguments={"action": "list"})],
    )
    res = parser.parse(resp)
    assert not res.is_finish
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "note_tool"
    assert res.tool_calls[0].arguments == {"action": "list"}


def test_parser_finish_tag():
    parser = ToolCallParser()
    resp = LLMResponse(content="Thought: Analyzed everything.\nFinish[这是给用户的完整答案]")
    res = parser.parse(resp)
    assert res.is_finish
    assert res.finish_answer == "这是给用户的完整答案"
    assert "Analyzed everything" in res.thought


def test_parser_text_tool_call_regex_and_ast():
    parser = ToolCallParser()
    # 模拟非严格 JSON（单引号、末尾逗号等 AST 可容错的情况）
    text = "Thought: 我需要写笔记\nAction: [TOOL_CALL: note_tool: {'action': 'create', 'title': '会议记录',}]"
    resp = LLMResponse(content=text)
    res = parser.parse(resp)
    assert not res.is_finish
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "note_tool"
    assert res.tool_calls[0].arguments["action"] == "create"
    assert res.tool_calls[0].arguments["title"] == "会议记录"


def test_parser_direct_text_as_finish():
    parser = ToolCallParser()
    resp = LLMResponse(content="你好！我是你的智能助手，有什么我可以帮你的？")
    res = parser.parse(resp)
    assert res.is_finish
    assert res.finish_answer == "你好！我是你的智能助手，有什么我可以帮你的？"
