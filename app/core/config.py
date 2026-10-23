from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------- LLM (支持纯文本与原生多模态输入) ----------
    llm_base_url: str = "http://127.0.0.1:8083/v1"
    llm_api_key: SecretStr = SecretStr("EMPTY")
    llm_model: str = "gemma-4-E4B-it-Q4_K_M.gguf"
    llm_use_native_tools: bool = True
    llm_temperature: float = 0.3
    llm_timeout: float = 120.0

    # ---------- Embedding ----------
    embed_provider: Literal["openai_compat", "dashscope", "local", "tfidf"] = "openai_compat"
    embed_base_url: str = "http://127.0.0.1:8084/v1"
    embed_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_model: str = "bge-small-zh-v1.5-f16.gguf"
    embed_dim: int = 512
    local_embed_model: str = "BAAI/bge-small-zh-v1.5"

    # ---------- Storage ----------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password123")
    redis_url: str = "redis://localhost:6379/0"
    sqlite_path: Path = Path("data/agent.db")
    data_dir: Path = Path("data")
    # 设为 true 跳过外部服务探测，直接用降级实现（测试/离线开发）
    offline_mode: bool = False

    # ---------- Agent / Context ----------
    max_steps: int = 8
    context_max_tokens: int = 8000
    context_reserve_ratio: float = 0.2
    history_window: int = 10
    reflect_every_n_steps: int = 3
    tool_timeout_seconds: float = 30.0
    tool_output_max_chars: int = 4000

    # ---------- Query Expansion ----------
    mqe_n: int = 3
    hyde_enabled: bool = True

    # ---------- RAG ----------
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    rag_top_k: int = 5
    # Minimum character-trigram containment for "quote -> textbook page" attribution.
    rag_attribute_min_overlap: float = 0.45

    # ---------- Memory ----------
    memory_top_k: int = 5
    working_ttl_seconds: int = 1800
    working_capacity: int = 50
    consolidate_working_threshold: float = 0.7
    consolidate_episodic_threshold: float = 0.8
    forget_importance_threshold: float = 0.2
    forget_max_age_days: int = 30
    forget_capacity: int = 1000
    lifecycle_interval_seconds: int = 600

    # ---------- Web / 临时知识库（Staging） ----------
    web_enabled: bool = False
    web_fetch_timeout: float = 15.0
    web_max_bytes: int = 2_000_000
    # 为空表示允许所有公网域名（内网/回环/元数据地址始终拒绝，除非允许回环用于测试）
    web_allowed_domains: list[str] = []
    web_allow_loopback: bool = False
    web_search_provider: Literal["none", "searxng", "tavily"] = "none"
    web_search_base_url: str = ""
    web_search_api_key: SecretStr | None = None
    web_search_timeout: float = 15.0
    # 主搜索服务不可用时的兜底：bing（内置 RSS 搜索，无需额外安装）/ none
    web_search_fallback_provider: Literal["none", "bing"] = "bing"
    # 可选 MCP 搜索服务（Streamable HTTP JSON-RPC），配置后优先于内置搜索 Provider。
    # 注意：这里只配置“MCP 搜索服务地址”，项目不内置 MCP 服务端。
    mcp_web_server_url: str = ""
    mcp_web_search_tool: str = "search"
    mcp_web_timeout: float = 20.0
    # 本地 RAG 不足时是否联网补充：off（默认）/ auto（后端自动）
    web_fallback_mode: Literal["off", "auto"] = "off"
    web_fallback_max_sources: int = 2
    web_fallback_min_similarity: float = 0.55
    web_fallback_min_hits: int = 2
    # 临时知识库
    staging_ttl_days: int = 7
    staging_max_documents_per_session: int = 30
    staging_max_text_chars: int = 40_000

    # ---------- Server ----------
    # 默认仅允许本地 llama.cpp WebUI 开发源，不再使用通配符 "*"。
    cors_origins: list[str] = ["http://127.0.0.1:5173", "http://localhost:5173"]
    rate_limit_per_minute: int = 60
    log_level: str = "INFO"
    # 可选 API Token：为空表示不鉴权（默认本机单用户用法）；配置后除 /health 外
    # 所有请求需携带 `Authorization: Bearer <token>`。
    api_token: SecretStr | None = None

    # ---------- Tools ----------
    # 终端工具默认不注册。只有显式开启后才会挂载（即使开启也只允许只读白名单命令）。
    enable_terminal_tool: bool = False
    # 客户端本地图片路径只允许落在该目录内（相对 data_dir）。
    image_allowed_subdirs: list[str] = ["uploads"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
