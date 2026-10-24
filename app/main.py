"""FastAPI 应用程序工厂与入口。"""

from __future__ import annotations

import os
from pathlib import Path

# 确保本地服务 (127.0.0.1 / localhost) 绝不经过系统代理 (如 Clash/v2ray)，防止 502 Bad Gateway 与额外延迟
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.middleware import setup_middlewares
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.learn import router as learn_router
from app.api.routes.llama_compat import router as llama_compat_router
from app.api.routes.memory import router as memory_router
from app.api.routes.rag import router as rag_router
from app.api.routes.session import router as session_router
from app.api.routes.web import router as web_router
from app.core.config import Settings, get_settings
from app.core.lifespan import lifespan
from app.observability.logging import setup_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    setup_logging(level=cfg.log_level)

    app = FastAPI(
        title="Learning Agent Backend (Llama.cpp + RAG + MCP)",
        description="辅助学习智能体后端服务：兼容 llama.cpp 原生 WebUI (/props, /v1/chat/completions, /mcp) + 学情记忆与 SM-2 间隔重复 + 知识图谱与教材 RAG",
        version="0.3.0",
        lifespan=lifespan,
    )

    # 注册中间件
    setup_middlewares(app, cfg)

    # 注册路由
    app.include_router(health_router)
    app.include_router(llama_compat_router)
    app.include_router(chat_router)
    app.include_router(rag_router)
    app.include_router(memory_router)
    app.include_router(session_router)
    app.include_router(learn_router)
    app.include_router(web_router)

    # 可选：前端构建产物存在时由同一进程托管 WebUI（单端口部署）。
    # WebUI 使用 hash 路由，index.html 即可覆盖所有页面；API 路由先注册，优先匹配。
    ui_dist = Path(__file__).resolve().parent.parent / "llama-webui" / "tools" / "ui" / "dist"
    if ui_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    return app


app = create_app()
