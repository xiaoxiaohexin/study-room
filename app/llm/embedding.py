"""Embedding 抽象与实现：DashScope → 本地 sentence-transformers → Hashing（兜底，无需网络）。

注意：不同 embedder 的向量空间不兼容，因此集合名会带上 `embedder.tag` 后缀做隔离。
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.config import Settings
from app.observability.logging import get_logger

log = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    dim: int
    tag: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """基于字符 n-gram 的 HashingVectorizer（TF-IDF 风格的无状态兜底方案，支持中文）。"""

    def __init__(self, dim: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.tag = f"hash{dim}"
        self._vec = HashingVectorizer(
            n_features=dim, analyzer="char_wb", ngram_range=(1, 3), alternate_sign=False, norm="l2"
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        m = self._vec.transform(texts)
        return np.asarray(m.todense(), dtype=np.float32).tolist()


class DashScopeEmbedder:
    """通过 OpenAI 兼容接口（llama-server / vLLM / DashScope 等）调用向量模型。"""

    batch_size = 10

    def __init__(self, settings: Settings, base_url: str | None = None, api_key: str | None = None):
        import httpx
        from openai import AsyncOpenAI

        self.dim = settings.embed_dim
        self.model = settings.embed_model
        self.tag = f"ds{self.dim}"
        resolved_url = base_url or settings.embed_base_url or settings.dashscope_base_url
        resolved_key = (
            api_key
            or (settings.embed_api_key.get_secret_value() if settings.embed_api_key else None)
            or (settings.dashscope_api_key.get_secret_value() if settings.dashscope_api_key else None)
            or "EMPTY"
        )
        is_local = "127.0.0.1" in resolved_url or "localhost" in resolved_url
        self.client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            http_client=httpx.AsyncClient(trust_env=not is_local, timeout=10.0),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            resp = await self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )
            for d in sorted(resp.data, key=lambda d: d.index):
                vec = list(d.embedding)
                if len(vec) > self.dim:
                    vec = vec[: self.dim]
                out.append(vec)
        return out


class LocalSTEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # 可选依赖

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.tag = f"st{self.dim}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = await asyncio.to_thread(self.model.encode, texts, normalize_embeddings=True)
        return vecs.tolist()


async def build_embedder(settings: Settings) -> tuple[Embedder, str]:
    """按配置构建 embedder，失败时逐级降级。返回 (embedder, status)。"""
    if settings.embed_provider == "openai_compat" and not settings.offline_mode:
        try:
            emb = DashScopeEmbedder(settings, base_url=settings.embed_base_url)
            vecs = await emb.embed(["ping"])
            if vecs and len(vecs[0]) > 0:
                emb.dim = len(vecs[0])
                emb.tag = f"emb{emb.dim}"
            return emb, f"ok:openai_compat({settings.embed_model})"
        except Exception as e:  # noqa: BLE001
            log.warning("openai_compat embedder unavailable, falling back", error=str(e))
    if (
        settings.embed_provider == "dashscope"
        and settings.dashscope_api_key
        and settings.dashscope_api_key.get_secret_value()
        and not settings.offline_mode
    ):
        try:
            emb = DashScopeEmbedder(settings, base_url=settings.dashscope_base_url)
            await emb.embed(["ping"])
            return emb, "ok:dashscope"
        except Exception as e:  # noqa: BLE001
            log.warning("dashscope embedder unavailable, falling back", error=str(e))
    if settings.embed_provider in ("openai_compat", "dashscope", "local") and not settings.offline_mode:
        try:
            return LocalSTEmbedder(settings.local_embed_model), "ok:local"
        except Exception as e:  # noqa: BLE001
            log.warning("local embedder unavailable, falling back to hashing", error=str(e))
    status = "ok:hashing" if settings.embed_provider == "tfidf" else "degraded:hashing"
    return HashingEmbedder(settings.embed_dim), status
