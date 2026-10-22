"""RagTool：允许 Agent 在推理执行中主动按需查询知识库文档。"""

from __future__ import annotations

from typing import Any

from app.observability.logging import get_logger
from app.rag.retriever import RagRetriever
from app.tools.base import Tool, ToolContext, ToolResult

log = get_logger(__name__)


class RagTool(Tool):
    name = "rag_search_tool"
    description = "用于在知识库中检索相关文档段落。当需要查阅上传的背景知识或专业资料时调用此工具。"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词或问题"},
            "top_k": {"type": "integer", "description": "返回文档数量上限（默认 3）"},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: RagRetriever):
        self.retriever = retriever

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments.get("query", "").strip()
        # Namespace always comes from the session scope; the model cannot select a
        # different knowledge base (or probe arbitrary namespaces).
        namespace = (context.rag_namespace or "default").strip() or "default"
        top_k = int(arguments.get("top_k", 3))

        log.info("rag_tool_invoked", query=query, namespace=namespace, top_k=top_k)
        if not query:
            return ToolResult(ok=False, error="query cannot be empty")

        docs = await self.retriever.search([query], namespace=namespace, top_k=top_k)

        staged_docs = []
        try:
            staged_docs = await self.retriever.search_staging(
                [query], session_id=context.session_id, top_k=min(3, top_k)
            )
        except Exception as exc:
            log.warning("rag_tool_staging_search_failed", error=str(exc))

        parts = []
        for d in docs:
            hdr = f" ({' > '.join(d.headers)})" if d.headers else ""
            low_notice = " [低匹配度提示：可能与查询主题无关]" if d.score < 0.40 else ""
            parts.append(f"### 文档: {d.filename}{hdr} (匹配分: {d.score:.2f}){low_notice}\n{d.content}")

        if staged_docs:
            staged_parts = []
            for sd in staged_docs:
                if sd.score >= 0.30:
                    url = sd.metadata.get("url") or "网页来源"
                    staged_parts.append(
                        f"### [临时暂存网页] {sd.filename} (来源: {url} | 匹配分: {sd.score:.2f})\n"
                        f"{sd.content}\n"
                        f"> 💡 提示：此内容已抓取暂存在临时库。若信息准确无误，可调用 web_tool(action='promote', doc_id='{sd.doc_id}') 提纯入库。"
                    )
            if staged_parts:
                if parts:
                    parts.append("--- 以下为【临时知识库暂存匹配】（联网抓取待提纯数据） ---")
                parts.extend(staged_parts)

        if not parts:
            return ToolResult(
                ok=True,
                output=(
                    f"在知识库 [{namespace}] 及临时暂存库中未检索到与 '{query}' 相关的文档。"
                    "建议调用 web_tool 工具进行针对性联网搜索。"
                ),
            )

        return ToolResult(ok=True, output="\n\n---\n\n".join(parts))
