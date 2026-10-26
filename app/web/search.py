"""联网搜索 Provider：SearXNG（自建、推荐）、Tavily（API Key）、Bing RSS + 360 搜索 + Bilibili（免配置多源兜底）。

抓取（fetch）与搜索（search）相互独立：没有搜索服务不影响直接贴 URL 导入。
内置反投毒过滤与多源相关性重排，解决单搜索引擎在多关键词查询下退化为单字词典条目的问题。
"""

from __future__ import annotations

import html as html_module
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

import httpx

from app.core.config import Settings
from app.observability.logging import get_logger

log = get_logger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_TAG_RE = re.compile(r"<[^>]+>")

# 对话式指令噪音清洗正则
_CONVERSATIONAL_NOISE_RE = re.compile(
    r"(?:继续收集|请?帮我(?:搜索|查找|收集)|用\s*web[_\s]*tool\s*工具(?:搜索|查找)?|"
    r"直到知识库[^，。；,.;]*|包括但不限于|等情况也能[^，。；,.;]*|也能认出[^，。；,.;]*|"
    r"不认错(?:角色)?|的相关(?:消息|信息|资料|内容)|那?补充知识[：:]?)",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(r"[\s,，。！？；：、|/\\()（）\[\]【】\"“”'‘’<>《》+]+")

# 泛化修饰词（在压缩长查询用于回退检索时优先剥离，保留核心实体）
_GENERIC_FILLER_TOKENS = {
    "角色",
    "游戏",
    "游戏内信息",
    "官网",
    "官方",
    "版本信息",
    "服务器状态",
    "相关",
    "消息",
    "信息",
    "资料",
    "介绍",
    "详情",
    "图片",
    "头像",
    "动漫",
    "二次元",
    "动漫角色",
    "character",
    "game",
    "official",
    "info",
}

# 单字词典/字源投毒结果特征
_DICTIONARY_POISON_PATTERNS = (
    "（汉语汉字）",
    "（汉语文字）",
    "的拼音,",
    "的解释,",
    "的部首,",
    "的笔顺",
    "汉语国学",
    "汉语查",
    "漢典",
    "剑桥词典",
    "cambridge.org/",
    "global.bing.com/dict",
    "/zidian/",
    "/cidian/",
)


class WebSearchError(Exception):
    """搜索服务不可用/未配置/返回异常。"""


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""


class SearchProvider:
    name = "none"

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        raise NotImplementedError


def sanitize_search_query(query: str) -> str:
    """剥离自然语言对话指令前后缀，保留适合搜索引擎的关键词序列。"""
    raw = (query or "").strip()
    if not raw:
        return ""
    cleaned = _CONVERSATIONAL_NOISE_RE.sub(" ", raw)
    tokens = [t.strip("的了呢吗吧啊") for t in _SPLIT_RE.split(cleaned) if t.strip("的了呢吗吧啊")]
    if not tokens:
        return raw[:80]
    deduped = list(dict.fromkeys(tokens))
    return " ".join(deduped)[:100]


def extract_search_keywords(query: str) -> list[str]:
    """提取查询中的核心关键词（去重、去泛化虚词），用于相关性打分与投毒检测。"""
    sanitized = sanitize_search_query(query)
    tokens = [t for t in _SPLIT_RE.split(sanitized) if t]
    if not tokens:
        return [query.strip()] if query.strip() else []
    core = [t for t in tokens if t.lower() not in _GENERIC_FILLER_TOKENS and len(t) >= 2]
    return core if core else tokens


def compact_bing_query(query: str, max_tokens: int = 2) -> str:
    """将多关键词查询压缩为 1~2 个核心实体词，规避 cn.bing.com 对 >=3 词查询的单字投毒截断。"""
    keywords = extract_search_keywords(query)
    if not keywords:
        return (query or "").strip()[:40]
    return " ".join(keywords[: max(1, max_tokens)])


def is_poisoned_search_results(query: str, results: list[WebSearchResult]) -> bool:
    """检测搜索结果是否遭遇搜索引擎单字截断投毒或与多词查询完全脱节。"""
    if not results:
        return False
    q_stripped = (query or "").strip()
    # 如果用户本身就在查单个汉字的字典解释，则不算投毒
    if len(q_stripped) <= 2 and not any(ch.isspace() for ch in q_stripped):
        return False

    dict_hits = 0
    for r in results:
        combined = f"{r.title} {r.url} {r.snippet}".lower()
        if any(pat.lower() in combined for pat in _DICTIONARY_POISON_PATTERNS):
            dict_hits += 1

    if dict_hits >= max(1, len(results) // 2):
        return True

    keywords = extract_search_keywords(query)
    if len(keywords) >= 2:
        secondary = [k.lower() for k in keywords[1:]]
        matched_secondary = 0
        for r in results:
            text = f"{r.title} {r.snippet}".lower()
            if any(k in text for k in secondary):
                matched_secondary += 1
        if matched_secondary == 0:
            return True

    return False


def score_result_relevance(query: str, result: WebSearchResult) -> float:
    """计算单条搜索结果与查询关键词的相关性得分，用于多源合并重排与过滤。"""
    combined = f"{result.title} {result.url} {result.snippet}".lower()
    if any(pat.lower() in combined for pat in _DICTIONARY_POISON_PATTERNS):
        return -1.0
    if "image.so.com" in combined or "image.baidu.com" in combined:
        return -0.5

    keywords = extract_search_keywords(query)
    if not keywords:
        return 0.5

    title_lower = (result.title or "").lower()
    snippet_lower = (result.snippet or "").lower()
    score = 0.0
    matched_count = 0

    for idx, kw in enumerate(keywords):
        kw_l = kw.lower()
        # 第 2 个及之后的具体实体词权重更高
        weight = 1.6 if idx == 1 else (1.3 if idx == 0 else 1.0)
        in_title = kw_l in title_lower
        in_snippet = kw_l in snippet_lower
        if in_title or in_snippet:
            matched_count += 1
            if in_title:
                score += 2.0 * weight
            if in_snippet:
                score += 1.0 * weight

    # 若包含全部前两个核心实体词，给予联合命中奖励
    if len(keywords) >= 2:
        k0, k1 = keywords[0].lower(), keywords[1].lower()
        if (k0 in title_lower or k0 in snippet_lower) and (k1 in title_lower or k1 in snippet_lower):
            score += 3.5
        elif k1 not in title_lower and k1 not in snippet_lower:
            # 仅命中第 1 个泛词而未提及核心专有名词时，显著降权
            score *= 0.15

    # 百科/Wiki/专题攻略长文加分
    if any(
        domain in result.url.lower()
        for domain in (
            "baike.baidu.com",
            "huijiwiki.com",
            "fandom.com",
            "moegirl.org",
            "taptap.cn",
            "gamersky.com",
            "9game.cn",
            "zol.com.cn",
            "bilibili.com/read/",
            "17173.com",
            "3dmgame.com",
        )
    ):
        score += 0.8

    return round(score, 4)


def _connection_hint(service: str, target: str, exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return (
            f"无法连接 {service}（{target}）：服务未启动、端口错误或被防火墙拦截。"
            "请确认地址后重试；也可以配置 WEB_SEARCH_FALLBACK_PROVIDER=bing 使用内置搜索。"
        )
    if isinstance(exc, httpx.TimeoutException):
        return f"{service} 请求超时（{target}）：请检查网络或服务负载。"
    return f"{service} 请求失败（{target}）：{exc}"


class SearxngProvider(SearchProvider):
    name = "searxng"

    def __init__(
        self,
        base_url: str,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        if not self.base_url:
            raise WebSearchError(
                "未配置 SearXNG 地址（WEB_SEARCH_BASE_URL）："
                "请填写运行中的 SearXNG 地址，或改用 WEB_SEARCH_PROVIDER=bing"
            )
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, trust_env=False, transport=self._transport
            ) as client:
                resp = await client.get(
                    f"{self.base_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "language": "zh-CN",
                        "safesearch": 1,
                        "pageno": 1,
                    },
                    headers={"User-Agent": _UA},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403, 429):
                raise WebSearchError(
                    f"SearXNG 拒绝访问（HTTP {status}）：该实例可能未启用 JSON API"
                    "（settings.yml 的 search.formats 需包含 json）、限流或需要认证。"
                ) from exc
            raise WebSearchError(f"SearXNG 返回 HTTP {status}（{self.base_url}）") from exc
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
            raise WebSearchError(_connection_hint("SearXNG", self.base_url, exc)) from exc

        results: list[WebSearchResult] = []
        for item in (data.get("results") or [])[: max(1, top_k)]:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title") or url)[:200],
                    url=url,
                    snippet=str(item.get("content") or "")[:500],
                    provider=self.name,
                )
            )
        return results


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.tavily.com",
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        if not self.api_key:
            raise WebSearchError(
                "未配置 Tavily API Key（WEB_SEARCH_API_KEY）："
                "请填写 Key，或改用 WEB_SEARCH_PROVIDER=bing"
            )
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, trust_env=False, transport=self._transport
            ) as client:
                resp = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max(1, top_k),
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise WebSearchError(
                f"Tavily 返回 HTTP {exc.response.status_code}：请检查 API Key 与额度。"
            ) from exc
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
            raise WebSearchError(_connection_hint("Tavily", self.base_url, exc)) from exc

        results: list[WebSearchResult] = []
        for item in (data.get("results") or [])[: max(1, top_k)]:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=str(item.get("title") or url)[:200],
                    url=url,
                    snippet=str(item.get("content") or "")[:500],
                    provider=self.name,
                )
            )
        return results


class BingProvider(SearchProvider):
    """Bing RSS 搜索：无需自建服务或 API Key，作为 SearXNG/Tavily 的兜底。

    内置多关键词反投毒回退：当 cn.bing.com 对 >=3 个词的查询返回单字字典投毒结果时，
    自动用核心双实体词二次检索并按全量关键词重排。
    """

    name = "bing"
    endpoint = "https://www.bing.com/search"

    def __init__(
        self,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout = timeout
        self._transport = transport

    @staticmethod
    def _clean(text: str) -> str:
        plain = _TAG_RE.sub("", text or "")
        return html_module.unescape(plain).strip()

    async def _fetch_rss(
        self, client: httpx.AsyncClient, q: str, count: int
    ) -> list[WebSearchResult]:
        try:
            resp = await client.get(
                self.endpoint,
                params={"q": q, "format": "rss", "count": max(5, count)},
                headers={
                    "User-Agent": _UA,
                    "Accept": "application/rss+xml,text/xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
            resp.raise_for_status()
            xml_text = resp.text
        except httpx.HTTPStatusError as exc:
            raise WebSearchError(
                f"Bing 搜索返回 HTTP {exc.response.status_code}，可能被限流，请稍后重试。"
            ) from exc
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
            raise WebSearchError(_connection_hint("Bing", self.endpoint, exc)) from exc

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise WebSearchError("Bing 返回内容不是合法 RSS（可能被重定向到验证页）。") from exc

        results: list[WebSearchResult] = []
        for item in root.findall("./channel/item"):
            url = (item.findtext("link") or "").strip()
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=self._clean(item.findtext("title") or url)[:200],
                    url=url,
                    snippet=self._clean(item.findtext("description") or "")[:500],
                    provider=self.name,
                )
            )
        return results

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        # 单元测试注入 MockTransport 时保持单次原样请求语义
        if self._transport is not None:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=True,
                transport=self._transport,
            ) as client:
                raw_results = await self._fetch_rss(client, query, max(1, top_k))
                return raw_results[: max(1, top_k)]

        effective_q = sanitize_search_query(query) or query
        async with httpx.AsyncClient(
            timeout=self.timeout,
            trust_env=False,
            follow_redirects=True,
        ) as client:
            results = await self._fetch_rss(client, effective_q, max(5, top_k))

            # 若命中 Bing 多词单字投毒，使用核心双实体词自动重试
            if is_poisoned_search_results(query, results):
                compacted = compact_bing_query(query, max_tokens=2)
                if compacted and compacted != effective_q:
                    log.info(
                        "bing_poison_detected_retrying_compacted",
                        original_query=query,
                        compacted_query=compacted,
                    )
                    retry_results = await self._fetch_rss(client, compacted, max(8, top_k * 2))
                    if retry_results and not is_poisoned_search_results(compacted, retry_results):
                        results = retry_results

        # 过滤残留字典投毒项并按查询相关性排序
        filtered = [r for r in results if score_result_relevance(query, r) >= 0.0]
        if filtered:
            filtered.sort(key=lambda r: score_result_relevance(query, r), reverse=True)
            return filtered[: max(1, top_k)]
        return results[: max(1, top_k)]


class So360Provider(SearchProvider):
    """360 搜索（so.com）：支持多关键词长查询，且直接从 data-mdurl 提取真实目标 URL。"""

    name = "so360"
    endpoint = "https://www.so.com/s"

    def __init__(
        self,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout = timeout
        self._transport = transport

    @staticmethod
    def _clean(text: str) -> str:
        plain = _TAG_RE.sub(" ", text or "")
        plain = html_module.unescape(plain)
        return re.sub(r"\s+", " ", plain).strip()

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        effective_q = sanitize_search_query(query) or query
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=True,
                transport=self._transport,
            ) as client:
                resp = await client.get(
                    self.endpoint,
                    params={"q": effective_q},
                    headers={
                        "User-Agent": _UA,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    },
                )
                resp.raise_for_status()
                html_text = resp.text
        except httpx.HTTPError as exc:
            raise WebSearchError(_connection_hint("360搜索", self.endpoint, exc)) from exc

        blocks = re.findall(r'<li class="res-list"[^>]*>(.*?)</li>', html_text, re.DOTALL)
        results: list[WebSearchResult] = []
        for block in blocks:
            title_m = re.search(r"<h3[^>]*>.*?<a[^>]*>(.*?)</a>", block, re.DOTALL)
            if not title_m:
                continue
            title = self._clean(title_m.group(1))
            if not title:
                continue
            mdurl_m = re.search(r'data-mdurl="([^"]+)"', block)
            href_m = re.search(r'<h3[^>]*>.*?<a[^>]*href="([^"]+)"', block, re.DOTALL)
            raw_url = html_module.unescape(
                (mdurl_m.group(1) if mdurl_m else (href_m.group(1) if href_m else "")).strip()
            )
            if not raw_url or not raw_url.startswith(("http://", "https://")):
                continue
            if "image.so.com" in raw_url or "so.com/link" in raw_url:
                continue

            # 移除 script/style 再提取摘要
            cleaned_block = re.sub(
                r"<(?:script|style)[^>]*>.*?</(?:script|style)>",
                " ",
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            snippet = self._clean(cleaned_block)
            if snippet.startswith(title):
                snippet = snippet[len(title) :].strip(" -_—")
            results.append(
                WebSearchResult(
                    title=title[:200],
                    url=raw_url,
                    snippet=snippet[:500],
                    provider=self.name,
                )
            )
            if len(results) >= max(5, top_k * 2):
                break

        results.sort(key=lambda r: score_result_relevance(query, r), reverse=True)
        return results[: max(1, top_k)]


class BilibiliSearchProvider(SearchProvider):
    """Bilibili 专栏与视频搜索：补充二次元/游戏角色设定、皮肤/私服、剧情解析等社区高密度信息。"""

    name = "bilibili"
    endpoint = "https://api.bilibili.com/x/web-interface/search/type"

    def __init__(
        self,
        timeout: float = 12.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout = timeout
        self._transport = transport

    @staticmethod
    def _clean(text: str) -> str:
        plain = _TAG_RE.sub("", text or "")
        plain = html_module.unescape(plain)
        return re.sub(r"\s+", " ", plain).strip()

    async def search(self, query: str, top_k: int = 5) -> list[WebSearchResult]:
        effective_q = sanitize_search_query(query) or query
        compacted_q = compact_bing_query(effective_q, max_tokens=4)
        headers = {
            "User-Agent": _UA,
            "Referer": "https://search.bilibili.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        results: list[WebSearchResult] = []
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=True,
                headers=headers,
                transport=self._transport,
            ) as client:
                await client.get("https://www.bilibili.com/")
                # 1. 搜索专栏文章 (read/cv...)
                r_art = await client.get(
                    self.endpoint,
                    params={"search_type": "article", "keyword": compacted_q},
                )
                if r_art.status_code == 200:
                    art_items = (r_art.json().get("data") or {}).get("result") or []
                    for item in art_items[: max(3, top_k)]:
                        cv_id = item.get("id")
                        title = self._clean(str(item.get("title") or ""))
                        desc = self._clean(str(item.get("desc") or ""))
                        if cv_id and title:
                            results.append(
                                WebSearchResult(
                                    title=f"{title} - 哔哩哔哩专栏"[:200],
                                    url=f"https://www.bilibili.com/read/cv{cv_id}",
                                    snippet=desc[:500],
                                    provider=self.name,
                                )
                            )
                # 2. 搜索视频简介 (video)
                r_vid = await client.get(
                    self.endpoint,
                    params={"search_type": "video", "keyword": compacted_q},
                )
                if r_vid.status_code == 200:
                    vid_items = (r_vid.json().get("data") or {}).get("result") or []
                    for item in vid_items[: max(3, top_k)]:
                        bvid = str(item.get("bvid") or "").strip()
                        title = self._clean(str(item.get("title") or ""))
                        desc = self._clean(str(item.get("description") or ""))
                        if bvid and title and desc:
                            results.append(
                                WebSearchResult(
                                    title=f"{title} - 哔哩哔哩"[:200],
                                    url=f"https://www.bilibili.com/video/{bvid}/",
                                    snippet=desc[:500],
                                    provider=self.name,
                                )
                            )
        except Exception as exc:  # noqa: BLE001
            raise WebSearchError(_connection_hint("Bilibili搜索", self.endpoint, exc)) from exc

        results.sort(key=lambda r: score_result_relevance(query, r), reverse=True)
        return results[: max(1, top_k)]


def build_search_providers(settings: Settings) -> list[SearchProvider]:
    """按优先级构建搜索 Provider 链：主 Provider -> 可选多源免配置兜底（Bing + 360 + Bilibili）。"""
    providers: list[SearchProvider] = []
    primary = settings.web_search_provider
    if primary == "searxng":
        providers.append(SearxngProvider(settings.web_search_base_url, settings.web_search_timeout))
    elif primary == "tavily":
        api_key = (
            settings.web_search_api_key.get_secret_value()
            if settings.web_search_api_key
            else ""
        )
        providers.append(
            TavilyProvider(
                api_key, settings.web_search_base_url or "https://api.tavily.com"
            )
        )

    fallback: Literal["none", "bing"] = settings.web_search_fallback_provider
    if fallback == "bing" and primary != "bing":
        providers.append(BingProvider())
        providers.append(So360Provider())
        providers.append(BilibiliSearchProvider())
    return providers


def build_search_provider(settings: Settings) -> SearchProvider | None:
    """兼容旧调用：返回主 Provider（无则返回第一个可用 Provider）。"""
    providers = build_search_providers(settings)
    return providers[0] if providers else None
