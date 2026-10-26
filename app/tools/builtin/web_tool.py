"""联网工具（WebTool）：搜索/抓取网页并存入临时知识库，支持一键提纯晋升与重复查询防死循环保护。

该工具注册在 ToolRegistry 中，因此会同时出现在 /tools 与 /mcp 的 tools/list 里；
执行作用域（session/user/知识域）来自 ToolContext，与其它内置工具保持一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.config import Settings
from app.observability.logging import get_logger
from app.tools.base import Tool, ToolContext, ToolResult
from app.web.fetcher import WebFetchError
from app.web.search import WebSearchError, extract_search_keywords, sanitize_search_query
from app.web.staging import WebStagingError, WebStagingService

if TYPE_CHECKING:
    from app.web.distill import DistillationService
    from app.web.promotion import PromotionService

log = get_logger(__name__)

_DIVERSIFY_ASPECTS = (
    "外观 服装 私服 皮肤 设定",
    "背景故事 剧情 技能 定位",
    "攻略 详情 评测 介绍",
)


class WebTool(Tool):
    name = "web_tool"
    description = (
        "联网搜索/抓取网页资料，并存入临时知识库（可选直接提纯晋升到正式知识库）。"
        "仅在 WEB_ENABLED=true 时可用；返回内容为非权威外部资料，引用需标注 URL。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "fetch", "promote"],
                "description": (
                    "search（联网搜索并抓取正文摘要暂存）、"
                    "fetch（抓取指定 URL 并暂存）或 "
                    "promote（将当前会话暂存文档提纯晋升到正式知识库）"
                ),
            },
            "query": {"type": "string", "description": "搜索关键词（action=search，建议精炼为 2~4 个核心词）"},
            "url": {"type": "string", "description": "网页地址（action=fetch）"},
            "top_k": {"type": "integer", "description": "返回结果上限（默认 3）"},
            "stage": {
                "type": "boolean",
                "description": "是否把抓取到的网页写入临时知识库（默认 true）",
            },
            "promote": {
                "type": "boolean",
                "description": "是否在暂存后自动提纯并晋升写入正式知识库（默认 false）",
            },
            "doc_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要晋升的临时文档 ID 列表（action=promote 时可选，留空则晋升当前会话全部暂存文档）",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        staging: WebStagingService,
        settings: Settings,
        distiller: DistillationService | None = None,
        promoter: PromotionService | None = None,
    ):
        self.staging = staging
        self.settings = settings
        self.distiller = distiller
        self.promoter = promoter
        self._session_queries: dict[str, list[str]] = {}

    async def _auto_promote_docs(
        self,
        doc_ids: list[str],
        *,
        user_id: str,
        course_id: str,
        namespace: str,
    ) -> dict[str, Any] | None:
        if not doc_ids or not self.promoter:
            return None
        if self.distiller:
            for did in doc_ids:
                try:
                    await self.distiller.distill_document(did, user_id=user_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning("web_tool_auto_distill_skipped", doc_id=did, error=str(exc))
        try:
            return await self.promoter.promote(
                staging_doc_ids=doc_ids,
                target_course_id=course_id,
                target_namespace=namespace,
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("web_tool_auto_promote_failed", error=str(exc))
            return None

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if not self.settings.web_enabled:
            return ToolResult(ok=False, error="联网功能未开启（WEB_ENABLED=false）")

        action = arguments.get("action")
        stage = bool(arguments.get("stage", True))
        do_promote = bool(arguments.get("promote", False))
        top_k = int(arguments.get("top_k") or 3)
        session_id = context.session_id or ""
        user_id = context.user_id
        target_ns = (context.rag_namespace or "default").strip() or "default"
        target_course = (context.course_id or target_ns or "default").strip() or "default"

        try:
            if action == "search":
                raw_query = (arguments.get("query") or "").strip()
                if not raw_query:
                    return ToolResult(ok=False, error="query 不能为空")

                norm_q = sanitize_search_query(raw_query).lower()
                history = self._session_queries.setdefault(session_id or "_default", [])
                repeat_count = sum(1 for q in history if q == norm_q)
                history.append(norm_q)

                effective_query = raw_query
                repeat_banner = ""
                if repeat_count >= 1:
                    core_kws = extract_search_keywords(raw_query)[:2]
                    aspect = _DIVERSIFY_ASPECTS[(repeat_count - 1) % len(_DIVERSIFY_ASPECTS)]
                    if core_kws:
                        effective_query = f"{' '.join(core_kws)} {aspect}"
                    repeat_banner = (
                        f"\n\n⚠️ [防重复检索提示] 你已检索过相同查询「{raw_query}」({repeat_count + 1} 次)，"
                        f"系统已自动拓展检索维度为「{effective_query}」。"
                        "请直接综合下方已抓取的正文摘要回答用户问题，切勿继续重复调用相同搜索！"
                    )

                results = await self.staging.search(effective_query, top_k=max(1, top_k))
                if not results:
                    return ToolResult(ok=True, output="联网搜索未找到结果。" + repeat_banner)

                lines = [f"- {r.title}\n  {r.url}\n  {r.snippet}" for r in results]
                staged_ids: list[str] = []
                excerpts_blocks: list[str] = []
                promoted_info = None

                if stage:
                    try:
                        staged = await self.staging.search_and_stage(
                            effective_query,
                            session_id=session_id,
                            user_id=user_id,
                            top_k=max(1, min(3, top_k)),
                            origin="mcp_tool",
                            results=results,
                        )
                        staged_ids = [d["doc_id"] for d in staged if d.get("doc_id")]
                        for d in staged:
                            exc_txt = (d.get("excerpt") or "").strip()
                            if exc_txt:
                                d_title = d.get("title") or d.get("search_title") or d.get("filename") or "网页正文"
                                d_url = d.get("url") or ""
                                excerpts_blocks.append(
                                    f"#### 📄 抓取正文摘要：《{d_title}》 (doc_id={d.get('doc_id')})\n"
                                    f"来源: {d_url}\n{exc_txt[:1000]}"
                                )
                        if do_promote and staged_ids:
                            promoted_info = await self._auto_promote_docs(
                                staged_ids,
                                user_id=user_id,
                                course_id=target_course,
                                namespace=target_ns,
                            )
                    except (WebFetchError, WebStagingError, WebSearchError) as exc:
                        log.warning("web_tool_stage_failed", error=str(exc))

                excerpt_section = (
                    "\n\n### 已抓取网页正文摘要\n" + "\n\n".join(excerpts_blocks)
                    if excerpts_blocks
                    else ""
                )
                suffix = (
                    f"\n\n已存入临时知识库: {', '.join(staged_ids)}（待提纯晋升）"
                    if staged_ids
                    else "\n\n（未暂存到临时知识库）"
                )
                if promoted_info and promoted_info.get("promoted_documents"):
                    p_docs = [d["doc_id"] for d in promoted_info["promoted_documents"]]
                    p_kps = [k["name"] for k in promoted_info.get("promoted_knowledge_points") or []]
                    suffix += f"\n已自动晋升至正式知识库 [{target_ns}]: 文档 {', '.join(p_docs)}"
                    if p_kps:
                        suffix += f" | 知识节点: {', '.join(p_kps)}"

                return ToolResult(
                    ok=True,
                    output=(
                        f"联网搜索结果（{len(results)} 条）：\n"
                        + "\n".join(lines)
                        + excerpt_section
                        + suffix
                        + repeat_banner
                    ),
                    metadata={
                        "results": [r.__dict__ for r in results],
                        "staged_doc_ids": staged_ids,
                        "promoted": promoted_info,
                    },
                )

            if action == "fetch":
                url = (arguments.get("url") or "").strip()
                if not url:
                    return ToolResult(ok=False, error="url 不能为空")
                if not stage:
                    preview = await self.staging.preview_url(url)
                    return ToolResult(
                        ok=True,
                        output=(
                            f"网页《{preview['title']}》抓取成功（{preview['chars']} 字）\n"
                            f"URL: {preview['final_url']}\n\n{preview['preview'][:1500]}"
                        ),
                        metadata=preview,
                    )
                doc = await self.staging.stage_url(
                    url,
                    session_id=session_id,
                    user_id=user_id,
                    origin="mcp_tool",
                    target_course_id=target_course,
                )
                promoted_info = None
                if do_promote and doc.get("doc_id"):
                    promoted_info = await self._auto_promote_docs(
                        [doc["doc_id"]],
                        user_id=user_id,
                        course_id=target_course,
                        namespace=target_ns,
                    )
                excerpt = (doc.get("excerpt") or "").strip()[:1200]
                promote_note = ""
                if promoted_info and promoted_info.get("promoted_documents"):
                    promote_note = f"\n已同步提纯并晋升至正式知识库 [{target_ns}]。"
                return ToolResult(
                    ok=True,
                    output=(
                        f"已抓取《{doc.get('title')}》并存入临时知识库"
                        f"（doc_id={doc['doc_id']}，{doc.get('chunks')} 个切片，待提纯晋升）。"
                        f"{promote_note}\n\n{excerpt}"
                    ).strip(),
                    metadata=doc,
                )

            if action == "promote":
                if not self.promoter:
                    return ToolResult(ok=False, error="当前容器未配置晋升服务（PromotionService）")
                doc_ids = [str(x).strip() for x in (arguments.get("doc_ids") or []) if str(x).strip()]
                if not doc_ids:
                    docs = await self.staging.list_documents(
                        session_id=session_id or None, user_id=user_id, limit=20
                    )
                    doc_ids = [
                        d["doc_id"]
                        for d in docs
                        if d.get("doc_id") and d.get("status") != "promoted"
                    ]
                if not doc_ids:
                    return ToolResult(ok=True, output="当前会话暂无待晋升的临时文档。")
                promoted_info = await self._auto_promote_docs(
                    doc_ids,
                    user_id=user_id,
                    course_id=target_course,
                    namespace=target_ns,
                )
                if not promoted_info:
                    return ToolResult(ok=False, error="提纯晋升执行失败")
                p_docs = [d["doc_id"] for d in promoted_info.get("promoted_documents") or []]
                p_kps = [k["name"] for k in promoted_info.get("promoted_knowledge_points") or []]
                return ToolResult(
                    ok=True,
                    output=(
                        f"已将 {len(p_docs)} 篇临时文档提纯晋升至正式知识库 [{target_ns}]"
                        + (f"，生成知识节点：{', '.join(p_kps)}" if p_kps else "")
                        + "。"
                    ),
                    metadata=promoted_info,
                )
        except (WebFetchError, WebSearchError, WebStagingError) as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - 工具失败不应打断 Agent
            log.warning("web_tool_failed", action=action, error=str(exc))
            return ToolResult(ok=False, error=f"联网工具执行失败: {exc}")

        return ToolResult(ok=False, error=f"未知操作: {action}")
