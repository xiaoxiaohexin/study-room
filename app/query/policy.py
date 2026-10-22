"""Deterministic retrieval policy helpers shared by chat entry points."""

from __future__ import annotations

import re
from typing import Literal

RetrievalMode = Literal["none", "local"]

_TRAILING = r"(?:呀|啊|哈|呢|吧|哦|啦|了|嘛|喔|哟|~|～|。|！|!|，|,|\?|？|\s)*"
_SMALLTALK_TOKEN = (
    r"(?:"
    r"你好|您好|早上好|中午好|下午好|晚上好|嗨|哈喽|hello|hi|hey|"
    r"在吗|在不在|好的|好|嗯|哦|收到|明白|知道了|谢谢|多谢|感谢|"
    r"再见|拜拜|晚安|ok|okay|"
    r"你是谁|你能做什么|你会做什么|有什么功能|怎么用|帮助"
    r")"
)
_SMALLTALK_RE = re.compile(
    rf"^{_SMALLTALK_TOKEN}(?:[\s，,。！!？?~～]+{_SMALLTALK_TOKEN})*{_TRAILING}$",
    re.IGNORECASE,
)

_CONTINUATION_RE = re.compile(
    rf"^(?:继续|接着|接着说|继续讲|展开|详细点|然后呢|还有呢|再说说)"
    rf"(?:吧|啊|呀|呢|一下|点)?{_TRAILING}$",
    re.IGNORECASE,
)

_WEB_INTENT_RE = re.compile(
    r"(?:"
    r"联网|上网|搜索|搜一下|搜搜|查一下|查查|查一查|网页|网上|"
    r"最新|近期|实时|新闻|官网|官方|来源|链接|"
    r"价格|汇率|天气|版本|发布|"
    r"\bsearch\b|\bweb\b|\blatest\b|\btoday\b|\bcurrent\b|\bnews\b"
    r")",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://", re.IGNORECASE)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip())


def is_non_retrieval_chat(query: str) -> bool:
    """Return true for turns that should use history/model knowledge only."""
    text = normalize_query(query)
    if not text or len(text) > 40:
        return False
    return bool(_SMALLTALK_RE.fullmatch(text) or _CONTINUATION_RE.fullmatch(text))


def has_web_intent(query: str) -> bool:
    """Return true when the user explicitly asks for fresh or external evidence."""
    text = normalize_query(query)
    if not text:
        return False
    return bool(_URL_RE.search(text) or _WEB_INTENT_RE.search(text))


def is_web_query_eligible(query: str) -> bool:
    """Final guard before automatic web fallback."""
    text = normalize_query(query)
    if not text or is_non_retrieval_chat(text):
        return False
    return bool(re.search(r"[\w\u4e00-\u9fff]", text))
