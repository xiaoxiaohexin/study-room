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


class ApiTokenMiddleware(BaseHTTPMiddleware):
    """Optional bearer-token guard. Disabled when ``api_token`` is empty.

    This is defense-in-depth for exposing the local backend beyond 127.0.0.1;
    the default single-user local setup keeps working with no token.
    """

    _EXEMPT_PATHS = {"/health"}

    def __init__(self, app, token: str | None = None):
        super().__init__(app)
        self._token = (token or "").strip() or None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if (
            self._token is None
            or request.method == "OPTIONS"
            or request.url.path in self._EXEMPT_PATHS
        ):
            return await call_next(request)

        authorization = request.headers.get("Authorization", "")
        provided = ""
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if not provided:
            provided = request.headers.get("X-API-Token", "").strip()

        if provided != self._token:
            log.warning("api_token_rejected", path=request.url.path, method=request.method)
            err = ErrorResponse(
                code=401,
                error="Unauthorized",
                detail="Missing or invalid API token",
                request_id=request_id_var.get(),
            )
            return JSONResponse(status_code=401, content=err.model_dump())
        return await call_next(request)


def setup_middlewares(app: FastAPI, settings: Settings) -> None:
    # 1. 可选 API Token 鉴权（在 CORS 之前注册，使 CORS 位于最外层并能处理预检）
    token = settings.api_token.get_secret_value() if settings.api_token else None
    app.add_middleware(ApiTokenMiddleware, token=token)
    if token:
        log.info("api_token_authentication_enabled")

    # 2. CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. 请求上下文与日志追踪中间件
    app.add_middleware(RequestContextMiddleware)

    # 4. 统一参数验证异常拦截 (HTTP 422 -> 400 规范化)
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
