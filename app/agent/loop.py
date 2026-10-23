"""智能体执行循环（AgentLoop）：Thought-Action-Observation-Reflection 核心状态机。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.reflection import Reflector
from app.context.builder import ContextBuilder
from app.core.tokens import count_tokens
from app.learning.router import IntentRouter
from app.learning.store import LearningStore
from app.llm.base import LLMClient
from app.memory.manager import MemoryManager
from app.observability.dialogue_logger import DialogueLogger
from app.observability.logging import get_logger
from app.query.expansion import QueryExpander
from app.rag.retriever import RagRetriever
from app.schemas.events import AgentEvent, EventType
from app.schemas.messages import Message
from app.schemas.requests import ChatRequest
from app.session.state_manager import SessionState, SessionStateManager
from app.tools.base import ToolContext
from app.tools.parser import ToolCallParser
from app.tools.registry import ToolRegistry

log = get_logger(__name__)


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        parser: ToolCallParser,
        context_builder: ContextBuilder,
        query_expander: QueryExpander,
        memory_manager: MemoryManager,
        rag_retriever: RagRetriever,
        state_manager: SessionStateManager,
        reflector: Reflector,
        dialogue_logger: DialogueLogger,
        use_native_tools: bool = True,
        default_max_steps: int = 8,
        learning_store: LearningStore | None = None,
        intent_router: IntentRouter | None = None,
        hyde_skip_score: float = 0.75,
        scratchpad_ratio: float = 0.35,
        max_reflections: int = 2,
    ):
        self.llm = llm
        self.tools = tools
        self.parser = parser
        self.context_builder = context_builder
        self.query_expander = query_expander
        self.memory_manager = memory_manager
        self.rag_retriever = rag_retriever
        self.state_manager = state_manager
        self.reflector = reflector
        self.dialogue_logger = dialogue_logger
        self.use_native_tools = use_native_tools
        self.default_max_steps = default_max_steps
        self.learning_store = learning_store
        self.intent_router = intent_router
        self.hyde_skip_score = hyde_skip_score
        self.scratchpad_ratio = scratchpad_ratio
        self.max_reflections = max_reflections

    def _scratchpad_budget(self) -> int:
        """ReAct scratchpad 可用 Token 预算，避免多步工具调用撑爆本地模型 n_ctx。"""
        base = getattr(self.context_builder, "max_tokens", 8000)
        return max(512, int(base * self.scratchpad_ratio))

    @staticmethod
    def _trim_scratchpad(scratchpad: list[Message], budget: int) -> None:
        """压缩最早的 Observation，不破坏 assistant tool_call / tool 的配对关系。"""
        total = sum(count_tokens(m.content or "") for m in scratchpad)
        if total <= budget:
            return
        for limit in (600, 240, 120):
            for msg in scratchpad:
                if total <= budget:
                    return
                if msg.role != "tool":
                    continue
                content = msg.content or ""
                if len(content) <= limit:
                    continue
                before = count_tokens(content)
                msg.content = content[:limit] + "\n…[Observation truncated to fit context budget]"
                total -= before - count_tokens(msg.content)

    async def run(
        self,
        request: ChatRequest,
        session: SessionState,
    ) -> AsyncIterator[AgentEvent]:
        start_time = time.time()
        max_steps = request.max_steps or self.default_max_steps
        user_id = request.user_id
        session_id = session.session_id

        log.info(
            "agent_loop_started",
            user_id=user_id,
            session_id=session_id,
            max_steps=max_steps,
            query_len=len(request.message),
            image_count=len(request.images or []),
        )

        yield AgentEvent(type=EventType.SESSION, step=0, data={"session_id": session_id})

        # 0. 学习意图路由（决定教学模式、专属教学 Prompt 与精简工具子集）
        active_tool_names: list[str] | None = None
        base_system_prompt = AGENT_SYSTEM_PROMPT
        active_mode = request.mode or "chat"
        retrieval_mode = "local"
        expose_tools = True
        if self.intent_router is not None:
            decision = await self.intent_router.route(
                user_id=user_id,
                message=request.message,
                images=request.images,
                explicit_mode=request.mode,
            )
            active_mode = decision.mode
            base_system_prompt = decision.system_prompt
            active_tool_names = decision.tool_names
            retrieval_mode = decision.retrieval_mode
            expose_tools = decision.expose_tools
            # 让 RouteDecision 的步数预算真正生效（此前恒为配置默认值，审计：死配置）。
            if request.max_steps is None and decision.max_steps:
                max_steps = decision.max_steps
            yield AgentEvent(
                type=EventType.STATUS,
                step=0,
                data={
                    "stage": "intent_routing",
                    "mode": decision.mode,
                    "tools": decision.tool_names,
                    "retrieval_mode": decision.retrieval_mode,
                    "web_allowed": decision.allow_web_fallback,
                    "detail": f"学习模式路由: [{decision.mode}] ({decision.reason}) | 挂载工具: {', '.join(decision.tool_names)}",
                },
            )

        effective_rag_namespace = (
            (request.rag_namespace and request.rag_namespace.strip())
            or (session.rag_namespace and session.rag_namespace.strip())
            or "default"
        )
        effective_course_id = (
            (request.course_id and request.course_id.strip())
            or (session.course_id and session.course_id.strip())
            or None
        )

        # 1. 查询扩展（带图请求或判卷/复习/计划模式跳过文本扩展，减少本地模型排队延迟）
        queries = [request.message]
        hyde_doc = None
        skip_expansion_modes = {"grade", "quiz", "review", "plan", "homework"}
        if (
            request.enable_query_expansion
            and not request.images
            and retrieval_mode != "none"
            and active_mode not in skip_expansion_modes
        ):
            yield AgentEvent(
                type=EventType.STATUS,
                step=0,
                data={"stage": "query_expansion", "detail": "正在执行 MQE 查询扩展与 HyDE 语义分析..."},
            )
            # 初筛：原始 query 已高分命中时跳过昂贵的 HyDE LLM 调用（审计 P1）。
            best_first_pass = 0.0
            try:
                pre_hits = await self.rag_retriever.search(
                    [request.message], namespace=effective_rag_namespace, top_k=3
                )
                best_first_pass = max((float(d.score) for d in pre_hits), default=0.0)
            except Exception as e:  # noqa: BLE001 - 初筛失败不影响主流程
                log.warning("query_expansion_precheck_failed", error=str(e))
            log.info(
                "query_expansion_precheck",
                best_first_pass_score=round(best_first_pass, 4),
                hyde_skipped=best_first_pass >= self.hyde_skip_score,
            )
            expanded = await self.query_expander.expand(
                request.message, hyde=best_first_pass < self.hyde_skip_score
            )
            queries = expanded.all_queries
            hyde_doc = expanded.hypothetical_doc

        # 2. 多源信息并行召回 + 学习者画像获取
        yield AgentEvent(
            type=EventType.STATUS,
            step=0,
            data={"stage": "retrieval", "detail": "正在并行检索学情画像、用户记忆与知识库..."},
        )
        if retrieval_mode == "none":
            memories, docs, learner_profile = [], [], ""
            yield AgentEvent(
                type=EventType.STATUS,
                step=0,
                data={
                    "stage": "retrieval_skipped",
                    "detail": "当前回合不需要外部检索，仅使用会话历史与模型已有知识。",
                },
            )
        elif self.learning_store is not None:
            memories, docs, learner_profile = await asyncio.gather(
                self.memory_manager.retrieve_all(
                    queries, user_id=user_id, top_k=5, session_id=session_id
                ),
                self.rag_retriever.search(queries, namespace=effective_rag_namespace, top_k=5),
                self.learning_store.build_learner_profile_text(user_id=user_id),
            )
        else:
            memories, docs = await asyncio.gather(
                self.memory_manager.retrieve_all(
                    queries, user_id=user_id, top_k=5, session_id=session_id
                ),
                self.rag_retriever.search(queries, namespace=effective_rag_namespace, top_k=5),
            )
            learner_profile = ""

        # 准备工具描述与上下文
        from app.agent.prompts import TEXT_PROTOCOL_ADDON

        system_prompt = base_system_prompt
        if not self.use_native_tools and expose_tools:
            system_prompt += f"\n{TEXT_PROTOCOL_ADDON}\n【可用工具列表】\n{self.tools.describe_for_prompt(names=active_tool_names)}"

        scratchpad_messages: list[Message] = []
        dialogue_steps_log: list[dict[str, Any]] = []
        tool_error_count = 0
        call_counts: dict[str, int] = {}
        reflection_count = 0
        last_reflection_step = -10**9
        tool_ctx = ToolContext(
            user_id=user_id,
            session_id=session_id,
            rag_namespace=effective_rag_namespace,
            course_id=effective_course_id,
        )
        final_answer = ""
        is_finished = False

        for step in range(max_steps):
            log.info("agent_loop_step_start", step=step, session_id=session_id, mode=active_mode)

            # GSSC 构建上下文（若带有图片，直接组装进原生多模态 User Message）
            built_ctx = self.context_builder.build(
                system_prompt=system_prompt,
                user_query=request.message,
                history=session.history,
                memories=memories,
                docs=docs,
                hyde_doc=hyde_doc,
                user_images=request.images,
                learner_profile=learner_profile,
            )

            # 组装本轮会话的全部消息
            current_messages = list(built_ctx.messages) + scratchpad_messages

            # 调用 LLM（仅挂载当前学习模式所需的精简工具子集）
            specs = (
                self.tools.specs(names=active_tool_names)
                if self.use_native_tools and expose_tools
                else None
            )
            llm_resp = await self.llm.chat(current_messages, tools=specs)

            # 解析行动
            parsed = self.parser.parse(llm_resp)

            # 发送 Thought 事件
            if parsed.thought:
                yield AgentEvent(type=EventType.THOUGHT, step=step, data={"text": parsed.thought})

            step_record = {
                "step": step,
                "thought": parsed.thought,
                "tool_calls": [tc.model_dump() for tc in parsed.tool_calls],
                "finish": parsed.is_finish,
            }

            # 命中 Finish 判定
            if parsed.is_finish:
                final_answer = parsed.finish_answer or llm_resp.content.strip()
                is_finished = True
                log.info("agent_loop_finish_reached", step=step, answer_len=len(final_answer))

                # 模拟流式增量 Token 推送
                for i in range(0, len(final_answer), 15):
                    chunk = final_answer[i : i + 15]
                    yield AgentEvent(type=EventType.TOKEN, step=step, data={"text": chunk})
                    await asyncio.sleep(0.01)

                yield AgentEvent(
                    type=EventType.FINAL,
                    step=step,
                    data={"answer": final_answer, "finished": True},
                )
                dialogue_steps_log.append(step_record)
                break

            # 处理工具调用
            if parsed.tool_calls:
                # 记录 Assistant 消息
                scratchpad_messages.append(
                    Message(
                        role="assistant",
                        content=parsed.thought or "Calling tool",
                        tool_calls=parsed.tool_calls,
                    )
                )

                for tc in parsed.tool_calls:
                    call_key = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False, default=str)}"
                    call_counts[call_key] = call_counts.get(call_key, 0) + 1

                    yield AgentEvent(
                        type=EventType.TOOL_CALL,
                        step=step,
                        data={"id": tc.id, "name": tc.name, "arguments": tc.arguments},
                    )

                    tool_res = await self.tools.execute(tc.name, tc.arguments, tool_ctx)
                    if not tool_res.ok:
                        tool_error_count += 1

                    yield AgentEvent(
                        type=EventType.OBSERVATION,
                        step=step,
                        data={"id": tc.id, "name": tc.name, "ok": tool_res.ok, "output": tool_res.output},
                    )

                    # 写入临时工作记忆 (重要性 0.6)
                    summary = f"执行工具 {tc.name} 结果: {tool_res.to_str()[:200]}"
                    await self.memory_manager.add(
                        type="working",
                        content=summary,
                        user_id=user_id,
                        session_id=session_id,
                        importance=0.6,
                    )

                    # 记录 Tool 响应消息
                    scratchpad_messages.append(
                        Message(
                            role="tool",
                            name=tc.name,
                            tool_call_id=tc.id,
                            content=tool_res.to_str(),
                        )
                    )

                step_record["observations"] = [m.content for m in scratchpad_messages if m.role == "tool"]
                # scratchpad 预算：多步工具调用时压缩早期观察，避免超出本地模型 n_ctx
                self._trim_scratchpad(scratchpad_messages, self._scratchpad_budget())

            # Reflection 反思评估：仅在出现异常信号时触发（工具报错 / 检索为空 / 重复调用）
            repeated_calls = sum(1 for v in call_counts.values() if v > 1)
            retrieval_empty = retrieval_mode != "none" and not docs and not memories
            if (
                request.enable_reflection
                and reflection_count < self.max_reflections
                and (step - last_reflection_step) >= self.reflector.interval_steps
                and self.reflector.should_reflect(
                    step,
                    tool_errors=tool_error_count,
                    retrieval_empty=retrieval_empty,
                    repeated_calls=repeated_calls,
                )
            ):
                reflection_count += 1
                last_reflection_step = step
                scratchpad_text = "\n".join(f"{m.role}: {m.content}" for m in scratchpad_messages[-6:])
                verdict = await self.reflector.reflect(request.message, scratchpad_text)
                yield AgentEvent(
                    type=EventType.REFLECTION,
                    step=step,
                    data={"done": verdict.done, "critique": verdict.critique, "next_hint": verdict.next_hint},
                )

                if verdict.done:
                    log.info("reflection_decided_done", step=step)
                    # 反思判定信息完备，要求 LLM 综合现有结果给出最终答案
                    synth_messages = (
                        current_messages
                        + scratchpad_messages
                        + [
                            Message(
                                role="user",
                                content="请根据以上收集到的全部信息，给出完整、结构化的最终答案。",
                            )
                        ]
                    )
                    synth_resp = await self.llm.chat(synth_messages)
                    final_answer = synth_resp.content.strip()
                    is_finished = True

                    for i in range(0, len(final_answer), 15):
                        yield AgentEvent(
                            type=EventType.TOKEN, step=step, data={"text": final_answer[i : i + 15]}
                        )
                        await asyncio.sleep(0.01)

                    yield AgentEvent(
                        type=EventType.FINAL,
                        step=step,
                        data={"answer": final_answer, "finished": True},
                    )
                    dialogue_steps_log.append(step_record)
                    break

            dialogue_steps_log.append(step_record)

        # 循环结束检查：若达到最大步数仍未结束
        if not is_finished:
            log.warning("agent_loop_max_steps_exceeded", max_steps=max_steps)
            yield AgentEvent(
                type=EventType.STATUS,
                step=max_steps,
                data={"stage": "max_steps", "detail": "已达到最大推理步数，正在基于现有信息整合回答..."},
            )
            force_prompt = (
                "已经达到最大推理步数。请基于目前所收集到的全部证据与思考过程，"
                "向用户提供尽可能完整客观的总结，并明确指出哪些部分可能需要进一步补充。"
            )
            final_resp = await self.llm.chat(
                current_messages + scratchpad_messages + [Message(role="user", content=force_prompt)]
            )
            final_answer = final_resp.content.strip()

            for i in range(0, len(final_answer), 15):
                yield AgentEvent(
                    type=EventType.TOKEN, step=max_steps, data={"text": final_answer[i : i + 15]}
                )
                await asyncio.sleep(0.01)

            yield AgentEvent(
                type=EventType.FINAL,
                step=max_steps,
                data={"answer": final_answer, "finished": False},
            )

        # 3. 持久化与记忆沉淀
        # 写入历史消息
        await self.state_manager.append_message(
            user_id=user_id,
            session_id=session_id,
            message=Message(role="user", content=request.message),
        )
        await self.state_manager.append_message(
            user_id=user_id,
            session_id=session_id,
            message=Message(role="assistant", content=final_answer),
        )

        # 沉淀到情景记忆（用户问答轮次，默认重要性 0.65）
        conv_summary = f"用户问：{request.message}\n助手答：{final_answer[:300]}"
        await self.memory_manager.add(
            type="episodic",
            content=conv_summary,
            user_id=user_id,
            session_id=session_id,
            importance=0.65,
        )
        if request.images:
            await self.memory_manager.add(
                type="perceptual",
                content=f"[图像交互] 用户问：{request.message}\n视觉回答摘要：{final_answer[:400]}",
                user_id=user_id,
                session_id=session_id,
                importance=0.75,
                metadata={"images": request.images},
            )

        # 触发记忆整合（严格限定当前会话，避免把其他会话的临时状态卷进来）
        await self.memory_manager.consolidate(user_id, session_id=session_id)

        # 记录全流程日志
        elapsed_sec = round(time.time() - start_time, 2)
        await self.dialogue_logger.log(
            {
                "user_id": user_id,
                "session_id": session_id,
                "query": request.message,
                "answer_len": len(final_answer),
                "steps": dialogue_steps_log,
                "elapsed_sec": elapsed_sec,
            }
        )

        yield AgentEvent(type=EventType.DONE, step=len(dialogue_steps_log), data={"elapsed_sec": elapsed_sec})
        log.info("agent_loop_completed", user_id=user_id, session_id=session_id, elapsed_sec=elapsed_sec)
