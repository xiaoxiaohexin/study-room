from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.agent.loop import AgentLoop
from app.agent.reflection import Reflector
from app.context.builder import ContextBuilder
from app.learning.router import IntentRouter
from app.llm.embedding import HashingEmbedder
from app.llm.fake import FakeLLM
from app.memory.episodic import EpisodicMemory
from app.memory.lifecycle import MemoryLifecycle
from app.memory.manager import MemoryManager
from app.memory.perceptual import PerceptualMemory
from app.memory.semantic import SemanticMemory
from app.memory.working import WorkingMemory
from app.observability.dialogue_logger import DialogueLogger
from app.query.expansion import QueryExpander
from app.rag.chunker import MarkdownChunker
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IngestionPipeline
from app.rag.retriever import RagRetriever
from app.schemas.events import EventType
from app.schemas.requests import ChatRequest
from app.session.state_manager import SessionStateManager
from app.storage.graph import NullGraphStore
from app.storage.sqlite import SQLiteStore
from app.storage.vector import InMemoryVectorStore
from app.tools.builtin.note_tool import NoteTool
from app.tools.parser import ToolCallParser
from app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_loop_multistep_execution(tmp_path: Path):
    db_file = tmp_path / "test_agent.db"
    sqlite = SQLiteStore(db_file)
    await sqlite.initialize()

    vec = InMemoryVectorStore()
    embedder = HashingEmbedder(dim=128)
    graph = NullGraphStore()

    wm = WorkingMemory(capacity=10)
    em = EpisodicMemory(sqlite=sqlite, vector_store=vec, embedder=embedder)
    sm = SemanticMemory(vector_store=vec, graph_store=graph, embedder=embedder)
    pm = PerceptualMemory()
    lifecycle = MemoryLifecycle(wm, em, sm)
    mm = MemoryManager(wm, em, sm, pm, lifecycle)

    loader = DocumentLoader()
    chunker = MarkdownChunker()
    IngestionPipeline(loader, chunker, embedder, vec)
    retriever = RagRetriever(vec, embedder)

    ctx_builder = ContextBuilder(max_tokens=4000)

    # 预设 FakeLLM 两步行为：
    # 步骤 1：调用 note_tool 创建笔记
    # 步骤 2：收到 Observation 后给出最终回答
    fake_llm = FakeLLM(
        responses=[
            'Thought: 需要创建笔记\nAction: [TOOL_CALL: note_tool: {"action": "create", "title": "测试", "content": "内容"}]',
            "Thought: 已经记录完毕。\nFinish[已成功为您创建笔记：测试]",
        ]
    )

    tools = ToolRegistry()
    notes_dir = tmp_path / "notes"
    tools.register(NoteTool(notes_root=notes_dir, sqlite=sqlite))

    parser = ToolCallParser()
    state_mgr = SessionStateManager(sqlite)
    reflector = Reflector(fake_llm)
    d_logger = DialogueLogger(tmp_path / "logs")
    query_expander = QueryExpander(fake_llm)

    loop = AgentLoop(
        llm=fake_llm,
        tools=tools,
        parser=parser,
        context_builder=ctx_builder,
        query_expander=query_expander,
        memory_manager=mm,
        rag_retriever=retriever,
        state_manager=state_mgr,
        reflector=reflector,
        dialogue_logger=d_logger,
        use_native_tools=False,
    )

    session = await state_mgr.get_or_create_session("u1")
    req = ChatRequest(user_id="u1", session_id=session.session_id, message="帮我记一条笔记", enable_query_expansion=False)

    events = []
    async for ev in loop.run(req, session):
        events.append(ev)

    event_types = [e.type for e in events]
    assert EventType.SESSION in event_types
    assert EventType.THOUGHT in event_types
    assert EventType.TOOL_CALL in event_types
    assert EventType.OBSERVATION in event_types
    assert EventType.FINAL in event_types
    assert EventType.DONE in event_types

    # 检查笔记是否真正落盘
    notes = list(notes_dir.rglob("*.md"))
    assert len(notes) == 1

    await sqlite.aclose()


@pytest.mark.asyncio
async def test_agent_loop_skips_retrieval_for_smalltalk(tmp_path: Path):
    sqlite = SQLiteStore(tmp_path / "smalltalk.db")
    await sqlite.initialize()
    state_mgr = SessionStateManager(sqlite)
    session = await state_mgr.get_or_create_session("u1")

    fake_llm = FakeLLM(responses=["Thought: 简单问候。\nFinish[你好，有什么可以帮你？]"])
    memory_manager = AsyncMock()
    retriever = AsyncMock()
    query_expander = AsyncMock()
    loop = AgentLoop(
        llm=fake_llm,
        tools=ToolRegistry(),
        parser=ToolCallParser(),
        context_builder=ContextBuilder(max_tokens=2000),
        query_expander=query_expander,
        memory_manager=memory_manager,
        rag_retriever=retriever,
        state_manager=state_mgr,
        reflector=Reflector(fake_llm),
        dialogue_logger=DialogueLogger(tmp_path / "logs"),
        use_native_tools=False,
        intent_router=IntentRouter(),
    )

    request = ChatRequest(
        user_id="u1",
        session_id=session.session_id,
        message="你好",
        enable_query_expansion=True,
    )
    events = [event async for event in loop.run(request, session)]

    retrieval_events = [
        event for event in events if event.type == EventType.STATUS and event.data.get("stage") == "retrieval_skipped"
    ]
    assert retrieval_events
    assert not [event for event in events if event.type == EventType.TOOL_CALL]
    retriever.search.assert_not_awaited()
    memory_manager.retrieve_all.assert_not_awaited()
    query_expander.expand.assert_not_awaited()

    await sqlite.aclose()
