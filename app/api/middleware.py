"""API 中间件：RequestID 上下文透传、统一错误包装、CORS 与访问限流。"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import Settings
from app.observability.logging import get_logger, request_id_var
from app.schemas.responses import ErrorResponse

log = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request_id_var.set(req_id)
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        log.info(
            "http_request_processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=req_id,
        )
        return response


def setup_middlewares(app: FastAPI, settings: Settings) -> None:
    # 1. CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. 请求上下文与日志追踪中间件
    app.add_middleware(RequestContextMiddleware)

    # 3. 统一参数验证异常拦截 (HTTP 422 -> 400 规范化)
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = request_id_var.get()
        log.warning("request_validation_error", errors=exc.errors(), request_id=req_id)
        err = ErrorResponse(
            code=400,
            error="Validation Error",
            detail=exc.errors(),
            request_id=req_id,
        )
        return JSONResponse(status_code=400, content=err.model_dump())

    # 4. 全局未捕获异常拦截 (HTTP 500)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = request_id_var.get()
        log.error("unhandled_server_exception", error=str(exc), request_id=req_id, exc_info=True)
        err = ErrorResponse(
            code=500,
            error="Internal Server Error",
            detail=str(exc),
            request_id=req_id,
        )
        return JSONResponse(status_code=500, content=err.model_dump())
