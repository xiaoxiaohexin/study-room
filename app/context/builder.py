"""GSSC 上下文流水线：Gather -> Select -> Structure -> Compress。"""

from __future__ import annotations

from app.context.packet import ContextPacket
from app.core.tokens import count_tokens, truncate_to_tokens
from app.memory.base import ScoredMemory
from app.observability.logging import get_logger
from app.rag.retriever import DocChunk
from app.schemas.messages import Message

log = get_logger(__name__)


class BuiltContext:
    def __init__(self, messages: list[Message], total_tokens: int, packet_count: int):
        self.messages = messages
        self.total_tokens = total_tokens
        self.packet_count = packet_count


class ContextBuilder:
    def __init__(self, max_tokens: int = 8000, reserve_ratio: float = 0.2):
        self.max_tokens = max_tokens
        self.reserve_ratio = reserve_ratio

    @property
    def budget_tokens(self) -> int:
        return int(self.max_tokens * (1.0 - self.reserve_ratio))

    def gather(
        self,
        system_prompt: str,
        user_query: str,
        history: list[Message],
        memories: list[ScoredMemory] | None = None,
        docs: list[DocChunk] | None = None,
        notes: list[str] | None = None,
        hyde_doc: str | None = None,
        learner_profile: str | None = None,
    ) -> list[ContextPacket]:
        """阶段1：多源信息汇聚。
        注意：hyde_doc 仅用于向量库检索（Query Expansion），不再注入 LLM Prompt，防止小参数模型将其误判为权威教材事实。
        """
        log.info(
            "context_gather_started",
            history_len=len(history),
            memories_len=len(memories or []),
            docs_len=len(docs or []),
            has_profile=bool(learner_profile),
        )
        packets: list[ContextPacket] = []

        # 1. 系统角色与提示词（必选，最高优先级）
        packets.append(
            ContextPacket(
                id="sys_prompt",
                source="system",
                content=system_prompt,
                tokens=count_tokens(system_prompt),
                relevance=1.0,
                recency=1.0,
                importance=1.0,
                priority=0,
            )
        )

        # 2. 当前用户问题（必选，次高优先级）
        packets.append(
            ContextPacket(
                id="user_query",
                source="history",
                content=user_query,
                tokens=count_tokens(user_query),
                relevance=1.0,
                recency=1.0,
                importance=1.0,
                priority=1,
            )
        )

        # 2.5 学习者画像与薄弱点摘要（高优先级）
        if learner_profile and learner_profile.strip():
            packets.append(
                ContextPacket(
                    id="learner_profile",
                    source="profile",
                    content=learner_profile.strip(),
                    tokens=count_tokens(learner_profile),
                    relevance=0.95,
                    recency=1.0,
                    importance=0.95,
                    priority=2,
                )
            )

        # 3. 历史对话
        for idx, msg in enumerate(history):
            text = f"{msg.role}: {msg.content or ''}"
            recency = (idx + 1) / (len(history) or 1)
            packets.append(
                ContextPacket(
                    id=f"hist_{idx}",
                    source="history",
                    content=text,
                    tokens=count_tokens(text),
                    relevance=0.8,
                    recency=recency,
                    importance=0.7,
                    priority=2,
                )
            )

        # 4. 检索到的记忆
        for sm in memories or []:
            text = f"[{sm.item.type.upper()} MEMORY] {sm.item.content}"
            packets.append(
                ContextPacket(
                    id=f"mem_{sm.item.id}",
                    source="memory",
                    content=text,
                    tokens=count_tokens(text),
                    relevance=sm.score,
                    recency=0.7,
                    importance=sm.item.importance,
                    priority=3,
                )
            )

        # 5. 检索到的知识库文档
        for d in docs or []:
            hdr = f" ({' > '.join(d.headers)})" if d.headers else ""
            page_label = getattr(d, "page_label", None)
            if not page_label:
                p_start = getattr(d, "page_start", 1)
                p_end = getattr(d, "page_end", p_start)
                page_label = f"P.{p_start}" if p_start == p_end else f"P.{p_start}-{p_end}"
            text = f"[DOCUMENT: {d.filename} {page_label}{hdr}]\n{d.content}"
            packets.append(
                ContextPacket(
                    id=f"doc_{d.id}",
                    source="rag",
                    content=text,
                    tokens=d.tokens or count_tokens(text),
                    relevance=d.score,
                    recency=0.6,
                    importance=0.8,
                    priority=3,
                )
            )

        # 6. 笔记记录
        for idx, note in enumerate(notes or []):
            packets.append(
                ContextPacket(
                    id=f"note_{idx}",
                    source="note",
                    content=f"[NOTE]\n{note}",
                    tokens=count_tokens(note),
                    relevance=0.75,
                    recency=0.8,
                    importance=0.8,
                    priority=3,
                )
            )

        log.info("context_gather_completed", total_packets=len(packets))
        return packets

    def select(self, packets: list[ContextPacket]) -> list[ContextPacket]:
        """阶段2：候选过滤、优先级排定与 Token 预算贪心选取。"""
        budget = self.budget_tokens
        mandatory = [p for p in packets if p.priority <= 1]
        candidates = [p for p in packets if p.priority > 1]

        used_tokens = sum(p.tokens for p in mandatory)
        selected = list(mandatory)

        # 按综合 score 降序排序
        candidates.sort(key=lambda p: (p.priority, -p.score))

        for cand in candidates:
            if used_tokens + cand.tokens <= budget:
                selected.append(cand)
                used_tokens += cand.tokens
            else:
                # 尝试局部截断容纳高相关度信息
                remaining = budget - used_tokens
                if remaining > 100:
                    truncated_content = truncate_to_tokens(cand.content, remaining)
                    cand.content = truncated_content + "\n...[TRUNCATED]"
                    cand.tokens = remaining
                    selected.append(cand)
                    used_tokens += remaining
                break

        log.info(
            "context_select_completed",
            budget=budget,
            used_tokens=used_tokens,
            selected_count=len(selected),
            dropped_count=len(packets) - len(selected),
        )
        return selected

    def structure_and_compress(
        self,
        selected: list[ContextPacket],
        user_query: str,
        user_images: list[str] | None = None,
    ) -> BuiltContext:
        """阶段3 & 4：结构化分区组装 [Role & Policies] / [Learner Profile] / [Evidence] / [Context] / [History]。"""
        role_parts = [p.content for p in selected if p.source == "system"]
        profile_parts = [p.content for p in selected if p.source == "profile"]
        evidence_parts = [p.content for p in selected if p.source in ("rag", "note")]
        context_parts = [p.content for p in selected if p.source in ("memory",)]
        history_parts = [p.content for p in selected if p.source == "history" and p.id != "user_query"]

        sections = []
        if role_parts:
            sections.append("### [Role & Policies]\n" + "\n\n".join(role_parts))

        if profile_parts:
            sections.append("### [Learner Profile & Mastery]\n" + "\n\n".join(profile_parts))

        if evidence_parts:
            sections.append("### [Evidence & Knowledge]\n" + "\n\n".join(evidence_parts))

        if context_parts:
            sections.append("### [Relevant Memories]\n" + "\n\n".join(context_parts))

        if history_parts:
            sections.append("### [Conversation History]\n" + "\n".join(history_parts))

        system_assembled = "\n\n---\n\n".join(sections)

        messages = [
            Message(role="system", content=system_assembled),
            Message(role="user", content=user_query, images=user_images),
        ]
        total_tokens = sum(count_tokens(m.content or "") for m in messages)

        log.info("context_structured", total_messages=len(messages), total_tokens=total_tokens)
        return BuiltContext(messages=messages, total_tokens=total_tokens, packet_count=len(selected))

    def build(
        self,
        system_prompt: str,
        user_query: str,
        history: list[Message],
        memories: list[ScoredMemory] | None = None,
        docs: list[DocChunk] | None = None,
        notes: list[str] | None = None,
        hyde_doc: str | None = None,
        user_images: list[str] | None = None,
        learner_profile: str | None = None,
    ) -> BuiltContext:
        packets = self.gather(
            system_prompt=system_prompt,
            user_query=user_query,
            history=history,
            memories=memories,
            docs=docs,
            notes=notes,
            hyde_doc=hyde_doc,
            learner_profile=learner_profile,
        )
        selected = self.select(packets)
        return self.structure_and_compress(selected, user_query, user_images=user_images)
