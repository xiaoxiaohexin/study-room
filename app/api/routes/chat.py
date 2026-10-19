"""Chat 对话路由：支持普通响应与 SSE 全双工流式推送。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
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
async def chat_sync(request: ChatRequest, container: AppContainer = Depends(get_container)):
    """非流式对话接口：收集全部推理步骤与最终回答后一次性返回。"""
    log.info("chat_sync_requested", user_id=request.user_id, session_id=request.session_id)
    session = await container.state_manager.get_or_create_session(
        user_id=request.user_id, session_id=request.session_id
    )

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
async def chat_stream(request: ChatRequest, container: AppContainer = Depends(get_container)):
    """SSE 流式事件接口：实时推送 Thought / ToolCall / Observation / Reflection / Token / Final 事件。"""
    log.info("chat_stream_requested", user_id=request.user_id, session_id=request.session_id)
    session = await container.state_manager.get_or_create_session(
        user_id=request.user_id, session_id=request.session_id
    )

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
