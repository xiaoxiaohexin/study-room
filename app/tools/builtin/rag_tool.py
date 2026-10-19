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
            "namespace": {"type": "string", "description": "知识库命名空间（默认 default）"},
            "top_k": {"type": "integer", "description": "返回文档数量上限（默认 3）"},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: RagRetriever):
        self.retriever = retriever

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = arguments.get("query", "").strip()
        namespace = arguments.get("namespace", "default")
        top_k = int(arguments.get("top_k", 3))

        log.info("rag_tool_invoked", query=query, namespace=namespace, top_k=top_k)
        if not query:
            return ToolResult(ok=False, error="query cannot be empty")

        docs = await self.retriever.search([query], namespace=namespace, top_k=top_k)
        if not docs:
            return ToolResult(ok=True, output=f"在知识库 [{namespace}] 中未检索到与 '{query}' 相关的文档。")

        parts = []
        for d in docs:
            hdr = f" ({' > '.join(d.headers)})" if d.headers else ""
            parts.append(f"### 文档: {d.filename}{hdr} (匹配分: {d.score:.2f})\n{d.content}")

        return ToolResult(ok=True, output="\n\n---\n\n".join(parts))
