"""依赖注入容器（AppContainer）：统一组装、管理各组件生命周期与平滑降级。"""

from __future__ import annotations

from pathlib import Path

from app.agent.loop import AgentLoop
from app.agent.reflection import Reflector
from app.context.builder import ContextBuilder
from app.core.config import Settings
from app.learning.router import IntentRouter
from app.learning.store import LearningStore
from app.llm.base import LLMClient
from app.llm.embedding import Embedder, build_embedder
from app.llm.fake import FakeLLM
from app.llm.openai_compat import OpenAICompatLLM
from app.memory.base import BaseMemory
from app.memory.episodic import EpisodicMemory
from app.memory.lifecycle import MemoryLifecycle
from app.memory.manager import MemoryManager
from app.memory.perceptual import PerceptualMemory
from app.memory.semantic import SemanticMemory
from app.memory.working import WorkingMemory
from app.observability.dialogue_logger import DialogueLogger
from app.observability.logging import get_logger
from app.query.expansion import QueryExpander
from app.rag.chunker import MarkdownChunker
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IngestionPipeline
from app.rag.retriever import RagRetriever
from app.session.state_manager import SessionStateManager
from app.storage.cache import Cache, InMemoryTTLCache, RedisCache
from app.storage.graph import GraphStore, Neo4jGraphStore, NullGraphStore
from app.storage.sqlite import SQLiteStore
from app.storage.vector import InMemoryVectorStore, QdrantVectorStore, SQLiteVectorStore, VectorStore
from app.tools.builtin.learning_tools import (
    GradeTool,
    KnowledgeTool,
    ProgressTool,
    QuizTool,
    ReviewTool,
)
from app.tools.builtin.memory_tool import MemoryTool
from app.tools.builtin.note_tool import NoteTool
from app.tools.builtin.rag_tool import RagTool
from app.tools.builtin.terminal_tool import TerminalTool
from app.tools.builtin.web_tool import WebTool
from app.tools.parser import ToolCallParser
from app.tools.registry import ToolRegistry
from app.web.distill import DistillationService
from app.web.fetcher import WebFetcher
from app.web.mcp_search import McpSearchClient
from app.web.promotion import PromotionService
from app.web.search import SearchProvider, build_search_providers
from app.web.staging import WebStagingService
from app.web.supplement import WebSupplementService

log = get_logger(__name__)


class AppContainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.status: dict[str, str] = {}

        # 1. 基础存储
        self.sqlite = SQLiteStore(settings.sqlite_path)
        self.vector_store: VectorStore
        self.graph_store: GraphStore
        self.cache: Cache

        # 2. 模型与嵌入（主模型直接采用多模态模型）
        self.llm: LLMClient
        self.embedder: Embedder

        # 3. 记忆系统与学习域核心
        self.working_memory: BaseMemory
        self.episodic_memory: BaseMemory
        self.semantic_memory: BaseMemory
        self.perceptual_memory: BaseMemory
        self.memory_lifecycle: MemoryLifecycle
        self.memory_manager: MemoryManager
        self.learning_store: LearningStore
        self.intent_router: IntentRouter

        # 4. RAG 系统
        self.doc_loader: DocumentLoader
        self.chunker: MarkdownChunker
        self.ingestion_pipeline: IngestionPipeline
        self.rag_retriever: RagRetriever

        # 4.5 联网 / 临时知识库（Staging）
        self.web_fetcher: WebFetcher
        self.web_search_providers: list[SearchProvider]
        self.mcp_web_client: McpSearchClient
        self.web_staging: WebStagingService
        self.web_distiller: DistillationService
        self.web_promoter: PromotionService
        self.web_supplement: WebSupplementService

        # 5. 上下文与查询扩充
        self.context_builder: ContextBuilder
        self.query_expander: QueryExpander

        # 6. 工具体系
        self.tool_registry: ToolRegistry
        self.tool_parser: ToolCallParser

        # 7. 会话状态与反思
        self.state_manager: SessionStateManager
        self.reflector: Reflector
        self.dialogue_logger: DialogueLogger

        # 8. 智能体循环核心
        self.agent_loop: AgentLoop

    async def initialize(self) -> None:
        log.info("initializing_container_components")
        s = self.settings

        # 1. 初始化 SQLite
        await self.sqlite.initialize()
        self.status["sqlite"] = "ok"

        # 2. 初始化 VectorStore（探活 Qdrant，未开启时默认采用 SQLite 持久化向量库）
        if not s.offline_mode:
            try:
                qdrant = QdrantVectorStore(
                    url=s.qdrant_url,
                    api_key=s.qdrant_api_key.get_secret_value() if s.qdrant_api_key else None,
                )
                await qdrant.ping()
                self.vector_store = qdrant
                self.status["vector_store"] = "ok:qdrant"
                log.info("vector_store_connected", backend="qdrant", url=s.qdrant_url)
            except Exception as e:  # noqa: BLE001
                log.warning("qdrant_unavailable_using_sqlite_vector", error=str(e))
                self.vector_store = SQLiteVectorStore(self.sqlite)
                self.status["vector_store"] = "ok:sqlite_vector"
        else:
            self.vector_store = InMemoryVectorStore()
            self.status["vector_store"] = "offline:in_memory"

        # 3. 初始化 GraphStore（探活 Neo4j，失败时降级为 NullGraphStore）
        if not s.offline_mode:
            try:
                neo = Neo4jGraphStore(
                    uri=s.neo4j_uri,
                    user=s.neo4j_user,
                    password=s.neo4j_password.get_secret_value(),
                )
                await neo.ping()
                self.graph_store = neo
                self.status["graph_store"] = "ok:neo4j"
                log.info("graph_store_connected", backend="neo4j", uri=s.neo4j_uri)
            except Exception as e:  # noqa: BLE001
                log.warning("neo4j_unavailable_degrading_to_null", error=str(e))
                self.graph_store = NullGraphStore()
                self.status["graph_store"] = "degraded:null"
        else:
            self.graph_store = NullGraphStore()
            self.status["graph_store"] = "offline:null"

        # 4. 初始化 Cache（探活 Redis，失败时降级为 InMemoryTTLCache）
        if not s.offline_mode:
            try:
                redis_c = RedisCache(s.redis_url)
                await redis_c.ping()
                self.cache = redis_c
                self.status["cache"] = "ok:redis"
                log.info("cache_connected", backend="redis", url=s.redis_url)
            except Exception as e:  # noqa: BLE001
                log.warning("redis_unavailable_degrading_to_inmemory", error=str(e))
                self.cache = InMemoryTTLCache()
                self.status["cache"] = "degraded:in_memory"
        else:
            self.cache = InMemoryTTLCache()
            self.status["cache"] = "offline:in_memory"

        # 5. 初始化 Embedder
        self.embedder, embed_status = await build_embedder(s)
        self.status["embedder"] = embed_status

        # 6. 初始化主模型 LLM（支持本地多模态 llama-server 或远程 OpenAI 兼容接口）
        has_key = bool(s.llm_api_key and s.llm_api_key.get_secret_value())
        if has_key and not s.offline_mode:
            self.llm = OpenAICompatLLM(s)
            self.status["llm"] = f"ok:{s.llm_model}"
            log.info("llm_client_initialized", model=s.llm_model, base_url=s.llm_base_url)
        else:
            self.llm = FakeLLM()
            self.status["llm"] = "fake:offline_test"
            log.info("llm_client_using_fake")

        # 7. 初始化记忆系统与学习域引擎
        self.working_memory = WorkingMemory(
            capacity=s.working_capacity,
            ttl_seconds=s.working_ttl_seconds,
        )
        self.episodic_memory = EpisodicMemory(
            sqlite=self.sqlite,
            vector_store=self.vector_store,
            embedder=self.embedder,
        )
        self.semantic_memory = SemanticMemory(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
        )
        self.perceptual_memory = PerceptualMemory(
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm=self.llm,
        )
        self.memory_lifecycle = MemoryLifecycle(
            working=self.working_memory,
            episodic=self.episodic_memory,
            semantic=self.semantic_memory,
            consolidate_working_thresh=s.consolidate_working_threshold,
            consolidate_episodic_thresh=s.consolidate_episodic_threshold,
            forget_importance_thresh=s.forget_importance_threshold,
            forget_max_age_days=s.forget_max_age_days,
            forget_capacity=s.forget_capacity,
        )
        self.memory_manager = MemoryManager(
            working=self.working_memory,
            episodic=self.episodic_memory,
            semantic=self.semantic_memory,
            perceptual=self.perceptual_memory,
            lifecycle=self.memory_lifecycle,
        )
        self.learning_store = LearningStore(self.sqlite, graph_store=self.graph_store)
        self.intent_router = IntentRouter(self.learning_store)
        self.status["learning_engine"] = "ok:sm2_ema"

        # 8. 初始化 RAG
        self.doc_loader = DocumentLoader(llm=self.llm)
        self.chunker = MarkdownChunker(
            chunk_size_tokens=s.chunk_size_tokens,
            chunk_overlap_tokens=s.chunk_overlap_tokens,
        )
        self.ingestion_pipeline = IngestionPipeline(
            loader=self.doc_loader,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            sqlite=self.sqlite,
        )
        # Backfill the RAG document registry so documents ingested before it existed
        # can still be previewed page by page.
        try:
            await IngestionPipeline.backfill_document_registry(
                self.sqlite, s.data_dir, self.embedder.tag
            )
        except Exception as e:  # noqa: BLE001 - backfill must never block startup
            log.warning("document_registry_backfill_failed", error=str(e))
        self.rag_retriever = RagRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

        # 8.5 初始化联网 / 临时知识库（Staging）与晋升管线
        self.web_fetcher = WebFetcher(s)
        self.web_search_providers = build_search_providers(s)
        self.mcp_web_client = McpSearchClient(s)
        self.web_staging = WebStagingService(
            settings=s,
            sqlite=self.sqlite,
            pipeline=self.ingestion_pipeline,
            fetcher=self.web_fetcher,
            search_providers=self.web_search_providers,
            mcp_client=self.mcp_web_client,
        )
        self.web_distiller = DistillationService(
            settings=s,
            sqlite=self.sqlite,
            llm=self.llm,
            chunker=self.chunker,
        )
        self.web_promoter = PromotionService(
            settings=s,
            sqlite=self.sqlite,
            pipeline=self.ingestion_pipeline,
            learning_store=self.learning_store,
            staging=self.web_staging,
        )
        self.web_supplement = WebSupplementService(settings=s, staging=self.web_staging)
        self.status["web"] = (
            "ok:enabled"
            if s.web_enabled
            else "disabled"
        )

        # 9. 初始化上下文与查询扩充
        self.context_builder = ContextBuilder(
            max_tokens=s.context_max_tokens,
            reserve_ratio=s.context_reserve_ratio,
        )
        self.query_expander = QueryExpander(llm=self.llm)

        # 10. 初始化工具体系（基础工具 + 5 个辅助学习专用工具）
        self.tool_registry = ToolRegistry(
            default_timeout=s.tool_timeout_seconds,
            max_output_chars=s.tool_output_max_chars,
        )
        notes_dir = Path(s.data_dir) / "notes"
        self.tool_registry.register(NoteTool(notes_root=notes_dir, sqlite=self.sqlite))
        # 终端工具默认不注册：避免通过 /tools、/mcp 触发任意命令执行（审计 P0-1）。
        if s.enable_terminal_tool:
            self.tool_registry.register(TerminalTool(workspace_root=s.data_dir))
            self.status["terminal_tool"] = "enabled:read_only_whitelist"
        else:
            self.status["terminal_tool"] = "disabled"
        self.tool_registry.register(MemoryTool(memory_manager=self.memory_manager))
        self.tool_registry.register(RagTool(retriever=self.rag_retriever))
        self.tool_registry.register(KnowledgeTool(store=self.learning_store))
        self.tool_registry.register(
            QuizTool(store=self.learning_store, retriever=self.rag_retriever, llm=self.llm)
        )
        self.tool_registry.register(GradeTool(store=self.learning_store, llm=self.llm))
        self.tool_registry.register(ReviewTool(store=self.learning_store))
        self.tool_registry.register(ProgressTool(store=self.learning_store))
        self.tool_registry.register(
            WebTool(
                staging=self.web_staging,
                settings=s,
                distiller=self.web_distiller,
                promoter=self.web_promoter,
            )
        )
        self.tool_parser = ToolCallParser()

        # 11. 初始化会话状态、反思与日志
        self.state_manager = SessionStateManager(sqlite=self.sqlite, history_window=s.history_window)
        self.reflector = Reflector(llm=self.llm, interval_steps=s.reflect_every_n_steps)
        self.dialogue_logger = DialogueLogger(log_dir=Path(s.data_dir) / "logs" / "dialogues")

        # 12. 组装智能体循环
        self.agent_loop = AgentLoop(
            llm=self.llm,
            tools=self.tool_registry,
            parser=self.tool_parser,
            context_builder=self.context_builder,
            query_expander=self.query_expander,
            memory_manager=self.memory_manager,
            rag_retriever=self.rag_retriever,
            state_manager=self.state_manager,
            reflector=self.reflector,
            dialogue_logger=self.dialogue_logger,
            use_native_tools=s.llm_use_native_tools,
            default_max_steps=s.max_steps,
            learning_store=self.learning_store,
            intent_router=self.intent_router,
        )
        log.info("container_initialization_completed", status=self.status)

    async def aclose(self) -> None:
        log.info("closing_container_resources")
        await self.sqlite.aclose()
        await self.vector_store.aclose()
        await self.graph_store.aclose()
        await self.cache.aclose()
        await self.llm.aclose()
        log.info("container_resources_closed")
