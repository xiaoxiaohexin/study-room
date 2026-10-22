"""联网抓取层：URL 校验（SSRF 防护，兼容本地 TUN Fake-IP）+ 受限下载 + 高信噪比 HTML 正文抽取。"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_module
import ipaddress
import re
import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx

from app.core.config import Settings
from app.observability.logging import get_logger

log = get_logger(__name__)

_TEXT_CONTENT_TYPES = ("text/html", "text/plain", "text/markdown", "application/xhtml+xml")

# 本地代理软件（Clash / Mihomo / Sing-box TUN 模式）标准 Fake-IP 映射网段 (RFC 2544: 198.18.0.0/15)
_TUN_FAKE_IP_NET = ipaddress.ip_network("198.18.0.0/15")

# 连同内部内容整块剔除的噪音标签（防止脚本/样式/导航栏污染正文前 1500 字摘要）
_STRIP_INNER_BLOCKS_RE = re.compile(
    r"<(?P<tag>script|style|noscript|svg|iframe|canvas|nav|header|footer|aside|form|template)"
    r"(?:\s[^>]*)?>.*?</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_MAIN_ARTICLE_RE = re.compile(
    r"<(?P<tag>article|main)(?:\s[^>]*)?>(.*?)</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_BLOCK_TAGS = re.compile(
    r"</?(?:script|style|noscript|svg|iframe|canvas|nav|header|footer|aside|form|"
    r"template|button|select|option|input|textarea)[^>]*>",
    re.IGNORECASE,
)
_BREAK_TAGS = re.compile(
    r"</?(?:p|div|br|li|ul|ol|tr|table|section|article|h[1-6]|blockquote|pre|hr)[^>]*>",
    re.IGNORECASE,
)
_ANY_TAG = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_CHARSET_RE = re.compile(r"charset=([\w-]+)", re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_NAV_LINK_LINE_RE = re.compile(r"^(?:\s*\*?\s*\[[^\]]{1,20}\]\([^)]+\)\s*[|｜/·•\s]*){3,}$")

_CAPTCHA_TITLES = ("百度安全验证", "安全验证", "访问验证", "Just a moment...", "Attention Required!")
_CAPTCHA_MARKERS = ("seccaptcha.baidu.com", "bfe_captcha", "BIOC_OPTIONS", "cf-chl-bypass")

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class WebFetchError(Exception):
    """抓取失败（校验、网络、超限、非文本内容等）。"""


@dataclass
class FetchedPage:
    url: str
    final_url: str
    title: str
    text: str
    content_hash: str
    fetched_at: float
    content_type: str = ""
    status: str = "ok"
    error: str = ""
    metadata: dict = field(default_factory=dict)


def _is_forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv4Address) and ip in _TUN_FAKE_IP_NET:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def normalize_url(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise WebFetchError("URL 不能为空")
    if not re.match(r"^https?://", text, re.IGNORECASE):
        text = "https://" + text
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise WebFetchError(f"不支持的 URL: {raw}")
    return text


async def validate_public_url(url: str, settings: Settings) -> str:
    """校验 URL 协议/域名/IP，阻止内网、回环、云元数据地址（SSRF 防护）。"""
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if not host:
        raise WebFetchError("URL 缺少主机名")

    allowed = [d.strip().lower() for d in (settings.web_allowed_domains or []) if d and d.strip()]
    host_allowed = any(host == d or host.endswith(f".{d}") for d in allowed)
    if allowed and not host_allowed and not (settings.web_allow_loopback and host in {"127.0.0.1", "localhost", "::1"}):
        raise WebFetchError(f"域名不在允许列表中: {host}")

    if settings.web_allow_loopback and host in {"127.0.0.1", "localhost", "::1"}:
        return normalized

    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise WebFetchError(f"域名解析失败: {host}") from exc

    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_forbidden_ip(ip):
            raise WebFetchError(f"拒绝访问内网/保留地址: {host} -> {ip}")
    return normalized


def _decode_body(raw: bytes, content_type: str) -> str:
    charset = ""
    match = _CHARSET_RE.search(content_type or "")
    if match:
        charset = match.group(1)
    for enc in (charset, "utf-8", "gb18030", "gbk", "latin-1"):
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="ignore")


def _post_clean_markdown(text: str) -> str:
    """清理 Markdown 转换后残留的埋点图片、纯导航链接行与多余空行。"""
    no_imgs = _MD_IMAGE_RE.sub("", text or "")
    lines: list[str] = []
    for raw_line in no_imgs.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _NAV_LINK_LINE_RE.match(line):
            continue
        # 过滤纯门户顶栏菜单行（短词 + 密集链接且无句读）
        if line.count("](") >= 4 and not any(p in line for p in ("。", "，", "：", "；")):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _html_to_text(html_text: str) -> str:
    """高信噪比正文抽取：先剥离 script/style/nav/header/footer 整块，再优先用 markitdown 或正则保底。"""
    pre_cleaned = _STRIP_INNER_BLOCKS_RE.sub(" ", html_text or "")

    # 若存在语义化 <article> 或 <main> 且内容充足，优先抽取核心正文区
    main_matches = _MAIN_ARTICLE_RE.findall(pre_cleaned)
    candidate_html = pre_cleaned
    if main_matches:
        joined_main = "\n".join(body for _, body in main_matches)
        plain_probe = _ANY_TAG.sub("", joined_main).strip()
        if len(plain_probe) >= 200:
            candidate_html = joined_main

    try:
        import io

        from markitdown import MarkItDown  # 可选依赖

        result = MarkItDown().convert_stream(
            io.BytesIO(candidate_html.encode("utf-8")), file_extension=".html"
        )
        text = _post_clean_markdown((result.text_content or "").strip())
        if len(text) >= 80 or (text and candidate_html is pre_cleaned):
            return text
    except Exception:  # noqa: BLE001 - fall back to regex extraction
        pass

    cleaned = _BLOCK_TAGS.sub(" ", candidate_html)
    cleaned = _BREAK_TAGS.sub("\n", cleaned)
    cleaned = _ANY_TAG.sub("", cleaned)
    cleaned = html_module.unescape(cleaned)
    cleaned = re.sub(r"[ \t\u3000]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    return _post_clean_markdown("\n".join(line for line in lines if line).strip())


def extract_relevant_excerpt(text: str, query: str = "", max_chars: int = 1800) -> str:
    """从网页正文中优先抽取与查询关键词高密度命中的段落摘要（保持原文顺序）。"""
    body = (text or "").strip()
    if not body:
        return ""
    if len(body) <= max_chars or not (query or "").strip():
        return body[:max_chars]

    from app.web.search import extract_search_keywords

    keywords = [k.lower() for k in extract_search_keywords(query) if k]
    if not keywords:
        return body[:max_chars]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if len(p.strip()) >= 15]
    if not paragraphs:
        return body[:max_chars]

    scored: list[tuple[float, int, str]] = []
    for idx, para in enumerate(paragraphs):
        p_low = para.lower()
        hits = sum(2.0 if i == 1 else 1.2 for i, kw in enumerate(keywords) if kw in p_low)
        # 给文章开头几段适度基础分以保留导语背景
        pos_bonus = 0.4 if idx < 3 else 0.0
        scored.append((hits + pos_bonus, idx, para))

    if max((s[0] for s in scored), default=0.0) <= 0.4:
        return body[:max_chars]

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected_indices: list[int] = []
    total_chars = 0
    for _score, idx, para in scored:
        if total_chars + len(para) + 2 > max_chars and selected_indices:
            continue
        selected_indices.append(idx)
        total_chars += len(para) + 2
        if total_chars >= max_chars:
            break

    selected_indices.sort()
    excerpt = "\n\n".join(paragraphs[i] for i in selected_indices)
    return excerpt[:max_chars] if excerpt else body[:max_chars]


def _extract_title(html_text: str, fallback_url: str) -> str:
    match = _TITLE_RE.search(html_text or "")
    if match:
        title = html_module.unescape(re.sub(r"\s+", " ", match.group(1))).strip()
        if title:
            return title[:120]
    parsed = urlparse(fallback_url)
    return (parsed.path.rstrip("/").split("/")[-1] or parsed.hostname or "网页资料")[:120]


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class WebFetcher:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._transport = transport

    async def _request(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[bytes, str, str]:
        """手动跟随重定向，并对每一跳做 SSRF 校验（防止 302 绕过）。"""
        current = url
        for _ in range(5):
            async with client.stream("GET", current, follow_redirects=False) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location") or ""
                    if not location:
                        raise WebFetchError("网页返回重定向但缺少 Location")
                    current = await validate_public_url(urljoin(current, location), self.settings)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not any(ct in content_type.lower() for ct in _TEXT_CONTENT_TYPES):
                    raise WebFetchError(f"不支持的内容类型: {content_type or 'unknown'}")
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.settings.web_max_bytes:
                        raise WebFetchError(
                            f"网页超过大小上限（{self.settings.web_max_bytes} bytes）"
                        )
                    chunks.append(chunk)
                return b"".join(chunks), content_type, str(response.url)
        raise WebFetchError("网页重定向次数过多")

    async def fetch(self, url: str) -> FetchedPage:
        if not self.settings.web_enabled:
            raise WebFetchError("联网功能未开启（WEB_ENABLED=false）")

        validated = await validate_public_url(url, self.settings)
        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=self.settings.web_fetch_timeout,
                trust_env=False,
                headers=_BROWSER_HEADERS,
                transport=self._transport,
            ) as client:
                raw, content_type, final_url = await self._request(client, validated)
        except WebFetchError:
            raise
        except httpx.HTTPStatusError as exc:
            raise WebFetchError(f"网页返回 HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise WebFetchError(f"网页请求失败: {exc}") from exc

        body = _decode_body(raw, content_type)
        if "html" in content_type.lower():
            title = _extract_title(body, final_url)
            if any(t in title for t in _CAPTCHA_TITLES) or any(
                m in body for m in _CAPTCHA_MARKERS
            ):
                raise WebFetchError(f"网页触发反爬安全验证（{title}）")
            text = _html_to_text(body)
        else:
            title = _extract_title("", final_url)
            text = body.strip()

        text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
        if not text:
            raise WebFetchError("网页没有抽取到有效正文（可能是纯脚本页面或需要登录）")
        if len(text) > self.settings.staging_max_text_chars:
            text = text[: self.settings.staging_max_text_chars]

        page = FetchedPage(
            url=normalized_input(url),
            final_url=final_url,
            title=title,
            text=text,
            content_hash=content_hash(text),
            fetched_at=time.time(),
            content_type=content_type,
        )
        log.info(
            "web_fetch_completed",
            url=page.final_url,
            title=page.title,
            chars=len(page.text),
        )
        return page


def normalized_input(url: str) -> str:
    try:
        return normalize_url(url)
    except WebFetchError:
        return (url or "").strip()
