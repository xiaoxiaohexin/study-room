"""Chat 对话路由：支持普通响应与 SSE 全双工流式推送。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_container
from app.core.container import AppContainer
from app.observability.logging import get_logger
from app.schemas.events import EventType
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse, StepRecord

log = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat_sync(
    request: ChatRequest,
    x_rag_namespace: str | None = Header(None, alias="X-RAG-Namespace"),
    x_course_id: str | None = Header(None, alias="X-Course-Id"),
    container: AppContainer = Depends(get_container),
):
    """非流式对话接口：收集全部推理步骤与最终回答后一次性返回。"""
    log.info("chat_sync_requested", user_id=request.user_id, session_id=request.session_id)

    # 1. 并发锁快速探测：若该会话已持有锁，直接返回 409 杜绝死锁与并发冲突
    if request.session_id:
        lock = container.state_manager.get_lock(request.session_id)
        if lock.locked():
            log.warning("concurrent_request_conflict", session_id=request.session_id)
            raise HTTPException(status_code=409, detail="A request is already running in this session.")

    raw_header_ns = unquote(x_rag_namespace) if x_rag_namespace else None
    clean_header_rag_ns = raw_header_ns.strip() if raw_header_ns and raw_header_ns.strip() else None
    clean_body_rag_ns = request.rag_namespace.strip() if request.rag_namespace and request.rag_namespace.strip() else None
    raw_header_cid = unquote(x_course_id) if x_course_id else None
    clean_header_course_id = raw_header_cid.strip() if raw_header_cid and raw_header_cid.strip() else None
    clean_body_course_id = request.course_id.strip() if request.course_id and request.course_id.strip() else None

    # 2. 读取已有会话或创建新会话（单轮对话不得篡改已有会话在 DB 中的持久化知识库绑定）
    session = None
    if request.session_id:
        session = await container.state_manager.get_session(user_id=request.user_id, session_id=request.session_id)
    if session is None:
        session = await container.state_manager.get_or_create_session(
            user_id=request.user_id,
            session_id=request.session_id,
            course_id=clean_body_course_id or clean_header_course_id,
            rag_namespace=clean_body_rag_ns or clean_header_rag_ns,
        )

    clean_session_rag_ns = (
        session.rag_namespace.strip()
        if session.rag_namespace and session.rag_namespace.strip()
        else None
    )
    clean_session_course_id = (
        session.course_id.strip()
        if session.course_id and session.course_id.strip()
        else None
    )

    effective_rag_ns = (
        clean_header_rag_ns
        or clean_body_rag_ns
        or clean_session_rag_ns
        or "default"
    )
    effective_course_id = (
        clean_header_course_id
        or clean_body_course_id
        or clean_session_course_id
        or None
    )
    request.rag_namespace = effective_rag_ns
    request.course_id = effective_course_id

    lock = container.state_manager.get_lock(session.session_id)
    if lock.locked():
        log.warning("concurrent_request_conflict", session_id=session.session_id)
        raise HTTPException(status_code=409, detail="A request is already running in this session.")

    async with lock:
        steps: list[StepRecord] = []
        curr_step = StepRecord(step=0)
        final_answer = ""
        finished = True

        async for ev in container.agent_loop.run(request, session):
            if ev.type == EventType.THOUGHT:
                curr_step.thought = ev.data.get("text", "")
            elif ev.type == EventType.TOOL_CALL:
                curr_step.tool = ev.data.get("name")
                curr_step.arguments = ev.data.get("arguments")
            elif ev.type == EventType.OBSERVATION:
                curr_step.observation = ev.data.get("output")
                steps.append(curr_step)
                curr_step = StepRecord(step=ev.step + 1)
            elif ev.type == EventType.FINAL:
                final_answer = ev.data.get("answer", "")
                finished = ev.data.get("finished", True)

        return ChatResponse(
            session_id=session.session_id,
            answer=final_answer,
            steps=steps,
            finished=finished,
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    x_rag_namespace: str | None = Header(None, alias="X-RAG-Namespace"),
    x_course_id: str | None = Header(None, alias="X-Course-Id"),
    container: AppContainer = Depends(get_container),
):
    """SSE 流式事件接口：实时推送 Thought / ToolCall / Observation / Reflection / Token / Final 事件。"""
    log.info("chat_stream_requested", user_id=request.user_id, session_id=request.session_id)

    # 1. 并发锁快速探测：若该会话已持有锁，直接返回 409 杜绝死锁与并发冲突
    if request.session_id:
        lock = container.state_manager.get_lock(request.session_id)
        if lock.locked():
            log.warning("concurrent_request_conflict", session_id=request.session_id)
            raise HTTPException(status_code=409, detail="A request is already running in this session.")

    raw_header_ns = unquote(x_rag_namespace) if x_rag_namespace else None
    clean_header_rag_ns = raw_header_ns.strip() if raw_header_ns and raw_header_ns.strip() else None
    clean_body_rag_ns = request.rag_namespace.strip() if request.rag_namespace and request.rag_namespace.strip() else None
    raw_header_cid = unquote(x_course_id) if x_course_id else None
    clean_header_course_id = raw_header_cid.strip() if raw_header_cid and raw_header_cid.strip() else None
    clean_body_course_id = request.course_id.strip() if request.course_id and request.course_id.strip() else None

    # 2. 读取已有会话或创建新会话（单轮对话不得篡改已有会话在 DB 中的持久化知识库绑定）
    session = None
    if request.session_id:
        session = await container.state_manager.get_session(user_id=request.user_id, session_id=request.session_id)
    if session is None:
        session = await container.state_manager.get_or_create_session(
            user_id=request.user_id,
            session_id=request.session_id,
            course_id=clean_body_course_id or clean_header_course_id,
            rag_namespace=clean_body_rag_ns or clean_header_rag_ns,
        )

    clean_session_rag_ns = (
        session.rag_namespace.strip()
        if session.rag_namespace and session.rag_namespace.strip()
        else None
    )
    clean_session_course_id = (
        session.course_id.strip()
        if session.course_id and session.course_id.strip()
        else None
    )

    effective_rag_ns = (
        clean_header_rag_ns
        or clean_body_rag_ns
        or clean_session_rag_ns
        or "default"
    )
    effective_course_id = (
        clean_header_course_id
        or clean_body_course_id
        or clean_session_course_id
        or None
    )
    request.rag_namespace = effective_rag_ns
    request.course_id = effective_course_id

    lock = container.state_manager.get_lock(session.session_id)
    if lock.locked():
        log.warning("concurrent_request_conflict", session_id=session.session_id)
        raise HTTPException(status_code=409, detail="A request is already running in this session.")

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        async with lock:
            try:
                async for ev in container.agent_loop.run(request, session):
                    yield ev.to_sse()
            except Exception as e:  # noqa: BLE001
                log.error("agent_loop_stream_error", error=str(e), exc_info=True)
                yield {"event": "error", "data": f'{{"error": "{e}"}}'}

    return EventSourceResponse(event_generator())
