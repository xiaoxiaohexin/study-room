"""针对联网架构优化与 RAG 虚高分修复的针对性测试用例。"""

import pytest

from app.rag.retriever import _calibrate_chunk_score, _extract_lexical_tokens
from app.web.search import (
    compact_bing_query,
    extract_search_keywords,
    is_poisoned_search_results,
    sanitize_search_query,
    WebSearchResult,
)


def test_sanitize_and_compact_query():
    raw_prompt = "继续收集莫宁的相关消息，直到知识库里面的数据能精准匹配出莫宁来，包括但不限于私服状态等情况也能认出该角色不认错"
    clean = sanitize_search_query(raw_prompt)
    assert "继续收集" not in clean
    assert "直到知识库" not in clean
    assert "莫宁" in clean

    keywords = extract_search_keywords("鸣潮 莫宁 角色 游戏 语音台词")
    assert "鸣潮" in keywords
    assert "莫宁" in keywords

    # Bing compacted query to avoid dictionary truncation
    compacted = compact_bing_query("鸣潮 莫宁 角色 游戏 语音台词")
    tokens = compacted.split()
    assert len(tokens) <= 2
    assert "鸣潮" in compacted or "莫宁" in compacted


def test_is_poisoned_search_results():
    poisoned = [
        WebSearchResult(
            title="鸣（汉语汉字）_百度百科",
            url="https://baike.baidu.com/item/%E9%B8%A3/10395499",
            snippet="又兽亦曰鸣。 《易·说卦传》：其於马也为善鸣...",
            provider="bing",
        ),
        WebSearchResult(
            title="鸣字的读音是míng，左边一个口右边一个鸟是什么字 - 汉语查",
            url="https://www.hgcha.com/zidian/be58baf2.html",
            snippet="鸣字的读音是míng，左边一个口右边一个鸟...",
            provider="bing",
        ),
    ]
    assert is_poisoned_search_results("鸣潮 莫宁 角色", poisoned) is True

    valid = [
        WebSearchResult(
            title="《鸣潮》共鸣者莫宁角色情报公开",
            url="https://www.taptap.cn/moment/12345",
            snippet="莫宁作为新版本登场的角色，拥有独特的机制...",
            provider="bing",
        )
    ]
    assert is_poisoned_search_results("鸣潮 莫宁 角色", valid) is False


def test_calibrate_chunk_score_penalizes_irrelevant_docs():
    queries = ["鸣潮 莫宁 角色 游戏 游戏内信息"]
    tokens = _extract_lexical_tokens(queries)
    assert "鸣潮" in tokens
    assert "莫宁" in tokens

    # 1. 模拟旧版误召回的 OpenAI 赛艇游戏 chunk
    unrelated_text = "励信号变高 的路径， 已经在第5章第5.7节详细讲过， 这里补充一个智能体语境 OpenAI 赛艇游戏在小圈反复撞分数点"
    # 低余弦相似度 0.32，即使 RRF 相对分是 1.0 (第 1 名)，也不能给出 1.00
    score_unrelated = _calibrate_chunk_score(
        chunk_text=unrelated_text,
        filename="llm-agent.pdf",
        query_tokens=tokens,
        raw_vector_score=0.32,
        rrf_relative_rank=1.0,
    )
    # 必须受到严重惩罚，不能误导大模型以为是 1.00
    assert score_unrelated < 0.40, f"Expected < 0.40, got {score_unrelated}"

    # 2. 模拟真正匹配的鸣潮莫宁文档
    relevant_text = "莫宁是《鸣潮》中的共鸣者角色，在私服与测试服中均有实装数据，属性为衍射..."
    score_relevant = _calibrate_chunk_score(
        chunk_text=relevant_text,
        filename="wuthering_waves_moning.md",
        query_tokens=tokens,
        raw_vector_score=0.82,
        rrf_relative_rank=1.0,
    )
    assert score_relevant > 0.85, f"Expected > 0.85, got {score_relevant}"
