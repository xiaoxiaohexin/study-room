"""查询理解与扩充模块：MQE（多样化子查询）与 HyDE（假设性文档生成）。"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from app.llm.base import LLMClient
from app.observability.logging import get_logger
from app.schemas.messages import Message

log = get_logger(__name__)


class ExpandedQuery(BaseModel):
    original: str
    sub_queries: list[str] = Field(default_factory=list)
    hypothetical_doc: str | None = None

    @property
    def all_queries(self) -> list[str]:
        queries = [self.original]
        queries.extend([q for q in self.sub_queries if q != self.original])
        if self.hypothetical_doc:
            queries.append(self.hypothetical_doc)
        return queries


MQE_PROMPT = """你是一个智能查询扩展助手。请针对用户的原始问题，生成 {n} 个语义等价但表达方式或侧重点不同的子查询，以便在知识库和记忆系统中更全面地召回相关信息。
请以 JSON 数组格式输出，不要包含多余的 Markdown 标记或解释。示例：["子查询1", "子查询2", "子查询3"]

原始问题：{query}"""

HYDE_PROMPT = """你是一个领域专家。请针对用户提出的问题，写一段简短、专业、信息密度高的假设性回答（100-200字即可），用作向量检索的基准锚点。即使你不知道确切事实，也请模拟一篇相关文档的论述风格。

用户问题：{query}"""


class QueryExpander:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._cache: dict[str, ExpandedQuery] = {}

    def _cache_key(self, query: str, n: int, hyde: bool) -> str:
        h = hashlib.md5(f"{query}:{n}:{hyde}".encode()).hexdigest()
        return h

    async def expand(self, query: str, mqe_n: int = 3, hyde: bool = True) -> ExpandedQuery:
        key = self._cache_key(query, mqe_n, hyde)
        if key in self._cache:
            log.debug("query_expansion_cache_hit", query=query)
            return self._cache[key]

        log.info("query_expansion_started", query=query, mqe_n=mqe_n, hyde=hyde)
        sub_queries: list[str] = []
        hypothetical_doc: str | None = None

        # 1. 执行 MQE
        if mqe_n > 0:
            try:
                prompt = MQE_PROMPT.format(n=mqe_n, query=query)
                resp = await self.llm.chat([Message(role="user", content=prompt)], temperature=0.5)
                cleaned = resp.content.strip().strip("`").replace("json\n", "").strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    sub_queries = [str(item).strip() for item in parsed if str(item).strip()]
                log.info("mqe_generated", sub_queries=sub_queries)
            except Exception as e:  # noqa: BLE001
                log.warning("mqe_generation_failed", error=str(e))

        # 2. 执行 HyDE
        if hyde:
            try:
                prompt = HYDE_PROMPT.format(query=query)
                resp = await self.llm.chat([Message(role="user", content=prompt)], temperature=0.3)
                hypothetical_doc = resp.content.strip()
                log.info("hyde_generated_length", char_count=len(hypothetical_doc))
            except Exception as e:  # noqa: BLE001
                log.warning("hyde_generation_failed", error=str(e))

        res = ExpandedQuery(
            original=query,
            sub_queries=sub_queries,
            hypothetical_doc=hypothetical_doc,
        )
        self._cache[key] = res
        return res
