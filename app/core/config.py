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

    # ---------- Server ----------
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 60
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
