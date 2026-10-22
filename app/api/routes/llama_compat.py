"""llama.cpp 原生后端 (llama-server) 与官方 SvelteKit WebUI 透明网关与 RAG/MCP 增强路由。

核心职责（对应四阶段改造方案）：
1. 反向代理监控与元数据路由：/props, /models, /v1/models, /slots, /tokenize, /detokenize, /reranking
   - 当本地 llama-server (默认 127.0.0.1:8083) 在线时，100% 原样透传；
   - 当本地 llama-server 未启动或离线测试时，提供与 llama-server 格式完全兼容的兜底响应，防止前端卡在 ServerLoadingSplash。
2. 拦截增强 POST /v1/chat/completions：
   - 在转发给 llama-server 之前，自动并行检索自定义后端的 RAG 教材切片、四层记忆与 SM-2 学情画像；
   - 采用前缀 KV Cache 友好的方式注入上下文（不破坏历史轮次 messages 前缀）；
   - 完整保留并透传 llama.cpp 专有采样参数（cache_prompt, min_p, top_k, top_p, repeat_penalty, repeat_last_n, samplers 等）；
   - 原生零缓冲透传 SSE 响应流（同时保留 delta.reasoning_content 思考流、delta.content 正文流与 timings 性能指标）。
3. 原生 MCP Server 协议端点 (POST /mcp)：
   - 将后端的 RAG 检索、课程知识图谱、SM-2 测验判卷、四层记忆等工具直接暴露为标准 MCP JSON-RPC 2.0 服务，
     可直接挂载到 llama.cpp WebUI 左侧的「MCP Servers」面板。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_container
from app.core.container import AppContainer
from app.core.tokens import count_tokens
from app.learning.router import RouteDecision
from app.observability.logging import get_logger
from app.schemas.messages import Message, ToolCall
from app.tools.base import ToolContext

log = get_logger(__name__)
router = APIRouter(tags=["Llama.cpp Native & WebUI Compat"])

# llama.cpp 默认扩展采样参数（当请求体未显式指定时自动补齐，与 llama-server /props 保持一致）
DEFAULT_LLAMA_SAMPLING: dict[str, Any] = {
    "cache_prompt": True,
    "min_p": 0.05,
    "top_k": 40,
    "top_p": 0.95,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "samplers": [
        "penalties",
        "dry",
        "top_n_sigma",
        "top_k",
        "typ_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature",
    ],
}


def _llama_origin(container: AppContainer) -> str:
    """从 settings.llm_base_url (例如 http://127.0.0.1:8083/v1) 提取 llama-server 根地址。"""
    base = container.settings.llm_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _extract_text_from_content(content: Any) -> str:
    """从 OpenAI / llama.cpp 消息的 content（字符串或多模态数组）中提取纯文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def _extract_images_from_content(content: Any) -> list[str]:
    """从 OpenAI / llama.cpp 消息的 content（字符串或多模态数组）中提取图片地址或 Base64 数据。"""
    if not isinstance(content, list):
        return []
    images: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "")
        if itype == "image_url":
            val = item.get("image_url")
            if isinstance(val, dict):
                url = val.get("url")
                if url:
                    images.append(str(url))
            elif isinstance(val, str) and val:
                images.append(val)
        elif itype in ("image", "input_image"):
            data = item.get("data") or item.get("image") or item.get("url")
            if data:
                images.append(str(data))
    return images


def _apply_intent_system_prompt(
    messages: list[dict[str, Any]],
    decision: RouteDecision,
) -> list[dict[str, Any]]:
    """将意图路由器确定的模式提示词融入 messages 中。
    保持系统前缀在同一会话模式下稳定，以最大化 KV Cache 命中。
    """
    cloned = [dict(m) for m in messages]
    sys_idx = -1
    for i, m in enumerate(cloned):
        if m.get("role") == "system":
            sys_idx = i
            break

    if sys_idx >= 0:
        orig = _extract_text_from_content(cloned[sys_idx].get("content", ""))
        if "【当前模式：" not in orig:
            if not orig or orig in ("You are a helpful assistant.", "You are a helpful assistant"):
                cloned[sys_idx]["content"] = decision.system_prompt
            else:
                cloned[sys_idx]["content"] = f"{decision.system_prompt}\n\n【用户偏好设置】\n{orig}"
    else:
        cloned.insert(0, {"role": "system", "content": decision.system_prompt})

    return cloned


async def _archive_dialogue_turn(
    container: AppContainer,
    user_id: str,
    session_id: str,
    query: str,
    answer: str,
    thought: str,
    tool_calls: list[dict[str, Any]],
    images: list[str],
    elapsed_sec: float,
    is_tool_followup: bool = False,
    course_id: str | None = None,
    rag_namespace: str | None = None,
) -> None:
    """统一沉淀会话日志 (JSONL)、四层学情记忆与会话历史状态到数据库。"""
    if not query and not answer:
        return

    effective_query = query or ("[用户上传了图片]" if images else "对话交互")

    # 1. 规范化 tool_calls 数据结构
    normalized_tool_calls: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = tc.get("name") or fn.get("name") or ""
        raw_args = tc.get("arguments") or fn.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:  # noqa: BLE001
                args = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        if name:
            normalized_tool_calls.append({"id": tc_id, "name": name, "arguments": args})

    # 2. 写入 dialogue_logger (输出至 data/logs/dialogues/YYYY-MM-DD.jsonl)
    step_record = {
        "step": 1 if is_tool_followup else 0,
        "thought": thought,
        "tool_calls": normalized_tool_calls,
        "finish": not normalized_tool_calls,
    }
    try:
        await container.dialogue_logger.log(
            {
                "user_id": user_id,
                "session_id": session_id,
                "query": effective_query,
                "answer_len": len(answer),
                "steps": [step_record],
                "elapsed_sec": elapsed_sec,
            }
        )
    except Exception as e:  # noqa: BLE001
        log.warning("dialogue_logger_record_failed", error=str(e))

    # 3. 写入 SessionStateManager (会话与历史消息隔离)
    try:
        sess = await container.state_manager.get_session(user_id=user_id, session_id=session_id)
        if sess is None:
            clean_ns = rag_namespace.strip() if rag_namespace and rag_namespace.strip() else "default"
            clean_cid = course_id.strip() if course_id and course_id.strip() else None
            await container.state_manager.get_or_create_session(
                user_id=user_id,
                session_id=session_id,
                course_id=clean_cid,
                rag_namespace=clean_ns,
            )
        if not is_tool_followup and effective_query:
            await container.state_manager.append_message(
                user_id=user_id,
                session_id=session_id,
                message=Message(role="user", content=effective_query),
            )
        if answer or normalized_tool_calls:
            msg_tool_objs = (
                [ToolCall(id=t["id"], name=t["name"], arguments=t["arguments"]) for t in normalized_tool_calls]
                if normalized_tool_calls
                else None
            )
            await container.state_manager.append_message(
                user_id=user_id,
                session_id=session_id,
                message=Message(
                    role="assistant",
                    content=answer or (thought if not answer else ""),
                    tool_calls=msg_tool_objs,
                ),
            )
    except Exception as e:  # noqa: BLE001
        log.warning("session_state_record_failed", error=str(e))

    # 4. 沉淀情景记忆与感知记忆
    try:
        if answer:
            await container.memory_manager.add(
                type="episodic",
                content=f"用户问：{effective_query}\n助手答：{answer[:300]}",
                user_id=user_id,
                session_id=session_id,
                importance=0.65,
            )
        if images and answer:
            img_refs = [
                img if len(img) < 256 else f"data:image/...({len(img)} bytes)"
                for img in images
            ]
            await container.memory_manager.add(
                type="perceptual",
                content=f"[图像交互] 用户问：{effective_query}\n视觉回答摘要：{answer[:400]}",
                user_id=user_id,
                session_id=session_id,
                importance=0.75,
                metadata={"images": img_refs},
            )
        # 触发记忆整合（仅当前会话作用域）
        await container.memory_manager.consolidate(user_id, session_id=session_id)
    except Exception as e:  # noqa: BLE001
        log.warning("memory_manager_archive_failed", error=str(e))


def _inject_context_into_messages(
    messages: list[dict[str, Any]],
    rag_block: str,
) -> list[dict[str, Any]]:
    """以 KV Cache 友好的方式将后端检索到的 RAG/记忆/学情注入到最后一条 user 消息中。

    这样可以保证多轮对话前面的 messages[0..-2] 完全不变，从而在 llama-server 开启
    `cache_prompt: true` 时 100% 命中历史轮次的 KV Cache！
    """
    if not rag_block.strip() or not messages:
        return messages

    cloned = [dict(m) for m in messages]
    # 找到最后一条 user 消息
    for idx in range(len(cloned) - 1, -1, -1):
        if cloned[idx].get("role") == "user":
            orig_content = cloned[idx].get("content", "")
            if isinstance(orig_content, str):
                cloned[idx]["content"] = (
                    f"{rag_block}\n\n"
                    f"### [User Question]\n{orig_content}"
                )
            elif isinstance(orig_content, list):
                new_parts = list(orig_content)
                injected = False
                for part_i, part in enumerate(new_parts):
                    if isinstance(part, dict) and part.get("type") == "text":
                        new_parts[part_i] = {
                            **part,
                            "text": f"{rag_block}\n\n### [User Question]\n{part.get('text', '')}",
                        }
                        injected = True
                        break
                if not injected:
                    new_parts.insert(0, {"type": "text", "text": rag_block})
                cloned[idx]["content"] = new_parts
            break
    return cloned


async def _build_rag_and_memory_block(
    container: AppContainer,
    user_query: str,
    user_id: str,
    namespace: str,
    session_id: str | None = None,
    retrieval_mode: str = "local",
    allow_web_fallback: bool = True,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """并行检索 RAG 教材切片、四层学情记忆与 SM-2 学习者画像，组装为精简上下文块。

    当本地 RAG 置信度不足且开启 ``WEB_FALLBACK_MODE=auto`` 时，会通过 MCP/搜索
    Provider 联网补充，并把来源暂存到临时知识库（B 通道）。返回
    ``(block, rag_sources, web_sources)``。
    """
    if not user_query:
        return "", [], []
    if retrieval_mode == "none":
        log.info(
            "retrieval_skipped",
            query_preview=user_query[:60],
            reason="route_policy",
        )
        return "", [], []

    s = container.settings
    tasks: list[Any] = [
        container.rag_retriever.search([user_query], namespace=namespace, top_k=s.rag_top_k),
        container.memory_manager.retrieve_all(
            [user_query], user_id=user_id, top_k=s.memory_top_k, session_id=session_id
        ),
    ]
    if container.learning_store is not None:
        tasks.append(container.learning_store.build_learner_profile_text(user_id=user_id))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    docs = results[0] if not isinstance(results[0], Exception) else []
    memories = results[1] if not isinstance(results[1], Exception) else []
    profile = (
        results[2]
        if len(results) > 2 and not isinstance(results[2], Exception)
        else ""
    )

    sections: list[str] = []
    rag_sources: list[dict[str, Any]] = []

    if profile and str(profile).strip():
        sections.append(f"### [Learner Profile & SM-2 Mastery]\n{str(profile).strip()}")

    if docs:
        doc_lines = []
        for d in docs:
            hdr = f" ({' > '.join(d.headers)})" if d.headers else ""
            page_label = getattr(d, "page_label", None)
            if not page_label:
                p_start = getattr(d, "page_start", 1)
                p_end = getattr(d, "page_end", p_start)
                page_label = f"P.{p_start}" if p_start == p_end else f"P.{p_start}-{p_end}"

            doc_lines.append(f"[DOCUMENT: {d.filename} {page_label}{hdr} | score={d.score:.2f}]\n{d.content}")
            rag_sources.append(
                {
                    "id": d.id,
                    "filename": d.filename,
                    "headers": d.headers,
                    "chunk_index": getattr(d, "chunk_index", 0),
                    "page_start": getattr(d, "page_start", 1),
                    "page_end": getattr(d, "page_end", 1),
                    "page_label": page_label,
                    "score": round(d.score, 3),
                    "content": d.content[:300] if hasattr(d, "content") else "",
                }
            )
        sections.append("### [Retrieved Course Materials (RAG)]\n" + "\n\n".join(doc_lines))

    if memories:
        mem_lines = [f"[{sm.item.type.upper()} MEMORY] {sm.item.content}" for sm in memories]
        sections.append("### [Relevant Learner Memories]\n" + "\n".join(mem_lines))

    # B 通道：本地知识库不足时联网补充，并自动暂存到临时知识库
    web_sources: list[dict[str, Any]] = []
    supplement = getattr(container, "web_supplement", None)
    if allow_web_fallback and supplement is not None and supplement.enabled:
        try:
            confidence = await container.rag_retriever.confidence([user_query], namespace=namespace)
            effective_local_count = sum(1 for d in docs if getattr(d, "score", 0.0) >= 0.40)
            if supplement.insufficient(confidence, effective_local_count, query=user_query):
                result = await supplement.supplement(
                    user_query,
                    session_id=session_id or "",
                    user_id=user_id,
                )
                if result.get("block"):
                    sections.append(str(result["block"]))
                    web_sources = list(result.get("sources") or [])
                    log.info(
                        "web_supplement_used",
                        namespace=namespace,
                        local_docs=len(docs),
                        max_similarity=round(float(confidence.get("max_similarity") or 0), 3),
                        web_sources=len(web_sources),
                        retrieval_mode=retrieval_mode,
                    )
        except Exception as exc:  # noqa: BLE001 - 联网补充失败必须降级为纯本地
            log.warning("web_supplement_failed", error=str(exc))

    if not sections:
        return "", [], []

    header_instruction = (
        "以下是系统从本地知识库（RAG）与学情记忆库中自动检索到的背景资料，"
        "请优先依据这些资料准确回答用户问题。"
        "若出现 [WEB: ...] 标记，那是联网补充的外部资料，属于非权威来源，"
        "引用时必须标注标题、URL 与抓取时间，且不得覆盖本地教材结论："
    )
    return header_instruction + "\n\n" + "\n\n".join(sections), rag_sources, web_sources


def _fallback_props(container: AppContainer) -> dict[str, Any]:
    """当本地 llama-server 未启动时，返回符合 llama-server 规范的 /props 结构，确保 WebUI 秒加载。"""
    s = container.settings
    return {
        "default_generation_settings": {
            "id": 0,
            "id_task": 0,
            "n_ctx": s.context_max_tokens,
            "speculative": False,
            "is_processing": False,
            "params": {
                "n_predict": -1,
                "seed": -1,
                "temperature": s.llm_temperature,
                "dynatemp_range": 0.0,
                "dynatemp_exponent": 1.0,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "top_n_sigma": -1.0,
                "xtc_probability": 0.0,
                "xtc_threshold": 0.1,
                "typical_p": 1.0,
                "repeat_last_n": 64,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "dry_multiplier": 0.0,
                "dry_base": 1.75,
                "dry_allowed_length": 2,
                "dry_penalty_last_n": -1,
                "mirostat": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "max_tokens": -1,
                "n_keep": 0,
                "n_discard": 0,
                "ignore_eos": False,
                "stream": True,
                "n_probs": 0,
                "min_keep": 0,
                "chat_format": "Content-only",
                "reasoning_format": "deepseek",
                "reasoning_in_content": False,
                "thinking_forced_open": False,
                "samplers": DEFAULT_LLAMA_SAMPLING["samplers"],
                "speculative.n_max": 16,
                "speculative.n_min": 0,
                "speculative.p_min": 0.75,
                "timings_per_token": False,
                "post_sampling_probs": False,
                "lora": [],
            },
            "prompt": "",
            "next_token": {
                "has_next_token": True,
                "has_new_line": False,
                "n_remain": -1,
                "n_decoded": 0,
                "stopping_word": "",
            },
        },
        "total_slots": 4,
        "model_alias": s.llm_model,
        "model_path": s.llm_model,
        "role": "model",
        "modalities": {
            "vision": True,
            "audio": False,
            "video": False,
        },
        "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
        "chat_template_caps": {
            "supports_parallel_tool_calls": True,
            "supports_preserve_reasoning": True,
            "supports_string_content": True,
            "supports_system_role": True,
            "supports_tool_calls": True,
            "supports_tools": True,
            "supports_typed_content": True,
        },
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "build_info": "b9010-xiaoxiao-rag-gateway",
        "is_sleeping": False,
        "cors_proxy_enabled": True,
    }


async def _proxy_to_llama(
    request: Request,
    container: AppContainer,
    upstream_path: str,
    fallback_fn=None,
) -> Response:
    """通用反向代理辅助函数：将请求原样转发至本地 llama-server，失败时调用 fallback_fn。"""
    origin = _llama_origin(container)
    url = f"{origin}{upstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "content-length"):
            headers[k] = v

    body = await request.body() if request.method in ("POST", "PUT", "PATCH", "DELETE") else None

    if not container.settings.offline_mode:
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=15.0) as client:
                resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body,
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={"Content-Type": resp.headers.get("content-type", "application/json")},
                )
        except Exception as e:  # noqa: BLE001
            log.debug("llama_upstream_unreachable_using_fallback", path=upstream_path, error=str(e))

    if fallback_fn is not None:
        data = fallback_fn()
        return JSONResponse(content=data)
    return JSONResponse(status_code=502, content={"error": f"Upstream llama-server ({origin}) unreachable"})


# =====================================================================
# 1. 元数据与监控路由反向代理 (/props, /models, /v1/models, /slots, /tokenize)
# =====================================================================


@router.get("/props")
@router.post("/props")
async def llama_props(request: Request, container: AppContainer = Depends(get_container)):
    """代理 llama-server 的 /props 接口（前端 ServerLoadingSplash 启动探活与默认采样参数必需）。"""
    return await _proxy_to_llama(
        request,
        container,
        "/props",
        fallback_fn=lambda: _fallback_props(container),
    )


@router.get("/models")
@router.get("/v1/models")
async def llama_models(request: Request, container: AppContainer = Depends(get_container)):
    """代理 llama-server 的 /models 与 /v1/models 模型列表接口。"""
    model_id = container.settings.llm_model

    def _fallback():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "name": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llamacpp",
                    "in_cache": True,
                    "path": model_id,
                    "status": {"value": "loaded", "args": ["llama-server", "--ctx-size", "8192"]},
                    "architecture": {"input_modalities": ["text", "image"]},
                    "context_length": container.settings.context_max_tokens,
                    "meta": {"vocab_type": 2, "n_vocab": 262144, "n_ctx_train": 131072},
                }
            ],
            "models": [
                {
                    "id": model_id,
                    "model": model_id,
                    "name": model_id,
                    "status": {"value": "loaded", "args": ["llama-server", "--ctx-size", "8192"]},
                    "capabilities": ["completion", "multimodal", "tools"],
                }
            ],
        }

    return await _proxy_to_llama(request, container, request.url.path, fallback_fn=_fallback)


@router.get("/slots")
@router.post("/slots/{id_slot}")
async def llama_slots(
    request: Request,
    id_slot: int | None = None,
    container: AppContainer = Depends(get_container),
):
    """代理 llama-server 的 /slots KV 槽位监控与缓存存取接口。"""
    path = f"/slots/{id_slot}" if id_slot is not None else "/slots"
    return await _proxy_to_llama(
        request,
        container,
        path,
        fallback_fn=lambda: [
            {
                "id": 0,
                "id_task": 0,
                "n_ctx": container.settings.context_max_tokens,
                "speculative": False,
                "is_processing": False,
            }
        ],
    )


@router.post("/tokenize")
async def llama_tokenize(request: Request, container: AppContainer = Depends(get_container)):
    """代理 llama-server 原生 /tokenize 分词接口，离线时回退到本地分词器。"""
    body_bytes = await request.body()

    def _fallback():
        try:
            payload = json.loads(body_bytes or b"{}")
            content = payload.get("content", "")
        except Exception:  # noqa: BLE001
            content = ""
        n = count_tokens(content)
        return {"tokens": list(range(n))}

    return await _proxy_to_llama(request, container, "/tokenize", fallback_fn=_fallback)


@router.post("/detokenize")
async def llama_detokenize(request: Request, container: AppContainer = Depends(get_container)):
    """代理 llama-server 原生 /detokenize 接口。"""
    return await _proxy_to_llama(request, container, "/detokenize", fallback_fn=lambda: {"content": ""})


@router.post("/reranking")
@router.post("/v1/rerank")
async def llama_reranking(request: Request, container: AppContainer = Depends(get_container)):
    """代理 llama-server 原生 /reranking 交叉编码器重排接口。"""
    return await _proxy_to_llama(request, container, "/reranking")


@router.post("/v1/embeddings")
@router.post("/embedding")
@router.post("/embeddings")
async def llama_embeddings(request: Request, container: AppContainer = Depends(get_container)):
    """代理或使用后端 Embedder 计算文本向量。"""
    payload = await request.json()
    inp = payload.get("input") or payload.get("content") or ""
    texts = [inp] if isinstance(inp, str) else list(inp)
    vecs = await container.embedder.embed(texts)
    return {
        "object": "list",
        "model": container.settings.embed_model,
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vecs)
        ],
        "usage": {"prompt_tokens": sum(count_tokens(t) for t in texts), "total_tokens": sum(count_tokens(t) for t in texts)},
    }


# =====================================================================
# 2. 核心：拦截 POST /v1/chat/completions 注入自定义 RAG 并对齐原生流式输出
# =====================================================================


@router.post("/v1/chat/completions")
async def llama_chat_completions(
    request: Request,
    container: AppContainer = Depends(get_container),
):
    """兼容 llama.cpp WebUI 与 OpenAI 标准的核心对话接口：
    1. 拦截请求，并行检索自定义后端的 RAG 知识库、四层学情记忆与 SM-2 画像；
    2. 将检索到的上下文以不破坏历史 KV Cache 的方式注入 messages；
    3. 完整保留并透传 llama.cpp 专有采样参数（cache_prompt, min_p, top_k, top_p, samplers 等）；
    4. 原样流式转发 llama-server 的 SSE 响应（完整保留 delta.reasoning_content 与 timings）。
    """
    body: dict[str, Any] = await request.json()
    stream: bool = bool(body.get("stream", False))
    raw_messages: list[dict[str, Any]] = body.get("messages", [])

    # 提取业务扩展参数（不污染发往 llama-server 的标准与采样字段）
    user_id: str = str(
        body.pop("user_id", None)
        or request.headers.get("X-User-Id")
        or "default_user"
    )
    header_rag_ns = request.headers.get("X-RAG-Namespace")
    body_rag_ns = body.pop("rag_namespace", None)
    header_course_id = request.headers.get("X-Course-Id")
    body_course_id = body.pop("course_id", None)
    session_id = str(body.pop("session_id", None) or request.headers.get("X-Conversation-Id") or "webui_default")

    raw_header_ns = unquote(header_rag_ns) if header_rag_ns else None
    clean_header_rag_ns = raw_header_ns.strip() if raw_header_ns and raw_header_ns.strip() else None
    clean_body_rag_ns = str(body_rag_ns).strip() if body_rag_ns and str(body_rag_ns).strip() else None
    raw_header_cid = unquote(header_course_id) if header_course_id else None
    clean_header_course_id = raw_header_cid.strip() if raw_header_cid and raw_header_cid.strip() else None
    clean_body_course_id = str(body_course_id).strip() if body_course_id and str(body_course_id).strip() else None

    session = None
    if session_id:
        session = await container.state_manager.get_session(user_id=user_id, session_id=session_id)
    if session is None:
        session = await container.state_manager.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            course_id=clean_body_course_id or clean_header_course_id,
            rag_namespace=clean_body_rag_ns or clean_header_rag_ns,
        )

    clean_session_rag_ns = (
        session.rag_namespace.strip()
        if session and session.rag_namespace and session.rag_namespace.strip()
        else None
    )
    clean_session_course_id = (
        session.course_id.strip()
        if session and session.course_id and session.course_id.strip()
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

    enable_rag: bool = bool(body.pop("enable_rag", True))
    explicit_mode: str | None = body.pop("learning_mode", None) or request.headers.get("X-Learning-Mode")

    # 判断是否为工具调用回调轮次 (role == "tool")
    is_tool_followup: bool = bool(raw_messages and raw_messages[-1].get("role") == "tool")

    # 提取最后一条用户提问文本与多模态图片
    last_user_query = ""
    images: list[str] = []
    for m in reversed(raw_messages):
        if m.get("role") == "user":
            last_user_query = _extract_text_from_content(m.get("content", ""))
            images = _extract_images_from_content(m.get("content", ""))
            break

    effective_query = last_user_query or ("[用户上传了图片]" if images else "")

    # 1. 意图路由判定（仅在非工具回调且有提问或图片时触发）
    decision: RouteDecision | None = None
    if not is_tool_followup and (effective_query or images):
        decision = await container.intent_router.route(
            user_id=user_id,
            message=effective_query,
            images=images if images else None,
            explicit_mode=explicit_mode,
        )

    # 2. 执行自定义后端的 RAG + 四层记忆 + SM-2 学情画像并行检索
    rag_block = ""
    rag_sources: list[dict[str, Any]] = []
    web_sources: list[dict[str, Any]] = []
    if enable_rag and effective_query and not is_tool_followup:
        retrieval_mode = decision.retrieval_mode if decision else "none"
        allow_web_fallback = decision.allow_web_fallback if decision else False
        rag_block, rag_sources, web_sources = await _build_rag_and_memory_block(
            container=container,
            user_query=effective_query,
            user_id=user_id,
            namespace=effective_rag_ns,
            session_id=session_id,
            retrieval_mode=retrieval_mode,
            allow_web_fallback=allow_web_fallback,
        )

    # 3. KV Cache 友好注入上下文与当前模式系统提示词
    augmented_messages = _inject_context_into_messages(raw_messages, rag_block)
    if decision:
        augmented_messages = _apply_intent_system_prompt(augmented_messages, decision)

    # 4. 根据当前模式精简暴露的工具集（防止 Gemma-4 E4B 工具选择过载）
    if decision and "tools" in body and isinstance(body["tools"], list):
        if not decision.expose_tools:
            body["tools"] = []
        elif decision.tool_names:
            allowed_tools = set(decision.tool_names)
            registered_tools = set(container.tool_registry._tools.keys())
            pruned_tools = [
                t
                for t in body["tools"]
                if (
                    t.get("function", {}).get("name") in allowed_tools
                    or t.get("function", {}).get("name") not in registered_tools
                )
            ]
            body["tools"] = pruned_tools

    # 5. 合并 llama.cpp 原生扩展采样参数（前端已传的参数优先，未传的补齐默认值）
    upstream_payload: dict[str, Any] = {
        **DEFAULT_LLAMA_SAMPLING,
        **body,
        "model": body.get("model") or container.settings.llm_model,
        "messages": augmented_messages,
        "stream": stream,
    }

    origin = _llama_origin(container)
    upstream_url = f"{origin}/v1/chat/completions"

    # 透传关键 Header（如 X-Conversation-Id 支持 llama-server 断点续传流）
    upstream_headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = container.settings.llm_api_key.get_secret_value()
    if api_key and api_key != "EMPTY":
        upstream_headers["Authorization"] = f"Bearer {api_key}"
    if "X-Conversation-Id" in request.headers:
        upstream_headers["X-Conversation-Id"] = request.headers["X-Conversation-Id"]

    log.info(
        "v1_chat_completions_intercepted",
        user_id=user_id,
        session_id=session_id,
        stream=stream,
        mode=decision.mode if decision else "followup",
        rag_hits=len(rag_sources),
        images_count=len(images),
        query_preview=effective_query[:60],
        retrieval_mode=decision.retrieval_mode if decision else "followup",
        web_allowed=decision.allow_web_fallback if decision else False,
    )

    start_time = time.time()

    # 3A. 流式模式（SSE 零延迟透传 delta.reasoning_content + delta.content + timings）
    if stream:
        async def _sse_generator() -> AsyncIterator[bytes]:
            collected_answer: list[str] = []
            collected_thought: list[str] = []
            collected_tool_calls_map: dict[int, dict[str, Any]] = {}
            used_upstream = False

            if not container.settings.offline_mode:
                try:
                    async with httpx.AsyncClient(trust_env=False, timeout=container.settings.llm_timeout) as client:
                        async with client.stream(
                            "POST",
                            upstream_url,
                            headers=upstream_headers,
                            json=upstream_payload,
                        ) as resp:
                            resp.raise_for_status()
                            used_upstream = True
                            async for line in resp.aiter_lines():
                                if not line:
                                    yield b"\n"
                                    continue
                                # 旁路收集正文、思考流与工具调用，不阻塞流式字节转发
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        chunk_obj = json.loads(line[6:])
                                        choices = chunk_obj.get("choices") or []
                                        if choices:
                                            delta = choices[0].get("delta") or {}
                                            if delta.get("reasoning_content"):
                                                collected_thought.append(str(delta["reasoning_content"]))
                                            if delta.get("content"):
                                                collected_answer.append(str(delta["content"]))
                                            if delta.get("tool_calls"):
                                                for tc in delta["tool_calls"]:
                                                    t_idx = tc.get("index", 0)
                                                    if t_idx not in collected_tool_calls_map:
                                                        collected_tool_calls_map[t_idx] = {
                                                            "id": tc.get("id") or f"call_{t_idx}",
                                                            "name": tc.get("function", {}).get("name", ""),
                                                            "arguments": "",
                                                        }
                                                    fn = tc.get("function", {})
                                                    if fn.get("name"):
                                                        collected_tool_calls_map[t_idx]["name"] = fn["name"]
                                                    if fn.get("arguments"):
                                                        collected_tool_calls_map[t_idx]["arguments"] += fn["arguments"]
                                    except Exception:  # noqa: BLE001
                                        pass
                                yield (line + "\n").encode("utf-8")
                except Exception as e:  # noqa: BLE001
                    log.warning("llama_upstream_stream_fallback", error=str(e))

            # 若本地 8083 未启动或处于离线测试模式，使用内置容器 LLM 生成标准 OpenAI SSE 流
            if not used_upstream:
                chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())
                model_name = upstream_payload["model"]
                msg_objs = [
                    Message(
                        role=m.get("role", "user"),
                        content=_extract_text_from_content(m.get("content", "")),
                    )
                    for m in augmented_messages
                ]
                async for ch in container.llm.stream(msg_objs):
                    delta_payload: dict[str, Any] = {}
                    if ch.reasoning_delta:
                        delta_payload["reasoning_content"] = ch.reasoning_delta
                        collected_thought.append(ch.reasoning_delta)
                    if ch.delta:
                        delta_payload["content"] = ch.delta
                        collected_answer.append(ch.delta)
                    sse_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta_payload,
                                "finish_reason": ch.finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(sse_chunk, ensure_ascii=False)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            # 异步沉淀全流程对话日志 (JSONL)、四层学情记忆与会话历史状态
            full_answer = "".join(collected_answer).strip()
            full_thought = "".join(collected_thought).strip()
            if not full_thought and full_answer:
                tm = re.search(r"<(?:think|Thought)>(.*?)</(?:think|Thought)>", full_answer, re.DOTALL)
                if tm:
                    full_thought = tm.group(1).strip()
            elapsed_sec = round(time.time() - start_time, 2)
            await _archive_dialogue_turn(
                container=container,
                user_id=user_id,
                session_id=session_id,
                query=effective_query,
                answer=full_answer,
                thought=full_thought,
                tool_calls=list(collected_tool_calls_map.values()),
                images=images,
                elapsed_sec=elapsed_sec,
                is_tool_followup=is_tool_followup,
                course_id=effective_course_id,
                rag_namespace=effective_rag_ns,
            )

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-RAG-Hits": str(len(rag_sources)),
                "X-Web-Sources": str(len(web_sources)),
                "X-Learning-Mode": decision.mode if decision else "followup",
                "X-Retrieval-Mode": decision.retrieval_mode if decision else "none",
            },
        )

    # 3B. 非流式模式
    if not container.settings.offline_mode:
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=container.settings.llm_timeout) as client:
                resp = await client.post(
                    upstream_url,
                    headers=upstream_headers,
                    json=upstream_payload,
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                c0 = choices[0] if choices else {}
                msg = c0.get("message") or {}
                ans = msg.get("content") or ""
                thought = msg.get("reasoning_content") or ""
                if not thought and ans:
                    tm = re.search(r"<(?:think|Thought)>(.*?)</(?:think|Thought)>", ans, re.DOTALL)
                    if tm:
                        thought = tm.group(1).strip()
                t_calls = msg.get("tool_calls") or []
                elapsed_sec = round(time.time() - start_time, 2)
                await _archive_dialogue_turn(
                    container=container,
                    user_id=user_id,
                    session_id=session_id,
                    query=effective_query,
                    answer=ans,
                    thought=thought,
                    tool_calls=t_calls,
                    images=images,
                    elapsed_sec=elapsed_sec,
                    is_tool_followup=is_tool_followup,
                    course_id=effective_course_id,
                    rag_namespace=effective_rag_ns,
                )
                return JSONResponse(
                    content=data,
                    headers={
                        "X-RAG-Hits": str(len(rag_sources)),
                        "X-Web-Sources": str(len(web_sources)),
                        "X-Learning-Mode": decision.mode if decision else "followup",
                        "X-Retrieval-Mode": decision.retrieval_mode if decision else "none",
                    },
                )
        except Exception as e:  # noqa: BLE001
            log.warning("llama_upstream_sync_fallback", error=str(e))

    # 离线/未启动 8083 时的非流式回退
    msg_objs = [
        Message(
            role=m.get("role", "user"),
            content=_extract_text_from_content(m.get("content", "")),
        )
        for m in augmented_messages
    ]
    llm_resp = await container.llm.chat(msg_objs)
    elapsed_sec = round(time.time() - start_time, 2)
    fallback_tc = (
        [tc.to_openai() for tc in llm_resp.tool_calls]
        if llm_resp.tool_calls
        else []
    )
    await _archive_dialogue_turn(
        container=container,
        user_id=user_id,
        session_id=session_id,
        query=effective_query,
        answer=llm_resp.content or "",
        thought=llm_resp.reasoning_content or "",
        tool_calls=fallback_tc,
        images=images,
        elapsed_sec=elapsed_sec,
        is_tool_followup=is_tool_followup,
        course_id=effective_course_id,
        rag_namespace=effective_rag_ns,
    )
    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": upstream_payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": llm_resp.content,
                        "reasoning_content": llm_resp.reasoning_content or None,
                    },
                    "finish_reason": llm_resp.finish_reason or "stop",
                }
            ],
            "usage": llm_resp.usage
            or {
                "prompt_tokens": count_tokens(last_user_query),
                "completion_tokens": count_tokens(llm_resp.content),
                "total_tokens": count_tokens(last_user_query) + count_tokens(llm_resp.content),
            },
        },
        headers={
            "X-RAG-Hits": str(len(rag_sources)),
            "X-Web-Sources": str(len(web_sources)),
            "X-Learning-Mode": decision.mode if decision else "followup",
            "X-Retrieval-Mode": decision.retrieval_mode if decision else "none",
        },
    )


# =====================================================================
# 3. 标准 MCP Server 端点 (/mcp)：供 llama.cpp WebUI「MCP Servers」直接挂载
# =====================================================================


@router.get("/mcp")
async def mcp_info(container: AppContainer = Depends(get_container)):
    """返回当前后端 MCP Server 元信息与暴露的工具清单。"""
    specs = container.tool_registry.specs()
    return {
        "name": "xiaoxiao-learning-rag-mcp",
        "version": "0.2.0",
        "protocol": "mcp-jsonrpc-2.0",
        "tools_count": len(specs),
        "tools": [
            {
                "name": s.name,
                "description": s.description,
                "inputSchema": s.parameters,
            }
            for s in specs
        ],
    }


async def _resolve_tool_scope(
    container: AppContainer,
    request: Request,
    user_id: str,
    session_id: str,
) -> tuple[str, str | None]:
    """Resolve the ``(rag_namespace, course_id)`` scope for tool execution.

    Precedence matches the chat routes: explicit header > session binding > default.
    """
    session = None
    if session_id:
        try:
            session = await container.state_manager.get_session(
                user_id=user_id, session_id=session_id
            )
        except Exception:  # noqa: BLE001 - session lookup must never break tool calls
            session = None

    raw_header_ns = request.headers.get("X-RAG-Namespace")
    header_ns = (unquote(raw_header_ns).strip() if raw_header_ns else "") or None
    session_ns = (session.rag_namespace or "").strip() if session and session.rag_namespace else None
    namespace = header_ns or session_ns or "default"

    raw_header_course = request.headers.get("X-Course-Id")
    header_course = (unquote(raw_header_course).strip() if raw_header_course else "") or None
    session_course = (session.course_id or "").strip() if session and session.course_id else None
    course_id = header_course or session_course or None
    return namespace, course_id


@router.post("/mcp")
async def mcp_jsonrpc_endpoint(
    request: Request,
    container: AppContainer = Depends(get_container),
):
    """标准 MCP (Model Context Protocol) Streamable HTTP / JSON-RPC 2.0 服务端点。
    可直接在 llama.cpp WebUI 左侧「MCP Servers」中添加 `http://127.0.0.1:8000/mcp` 挂载全部后端工具！
    """
    payload = await request.json()
    req_id = payload.get("id")
    method = payload.get("method", "")
    params = payload.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "xiaoxiao-learning-rag-mcp",
                    "version": "0.2.0",
                },
            },
        }

    if method in ("notifications/initialized", " initialized"):
        return Response(status_code=202)

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        specs = container.tool_registry.specs()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "inputSchema": s.parameters,
                    }
                    for s in specs
                ]
            },
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}
        user_id = str(arguments.pop("user_id", None) or request.headers.get("X-User-Id", "default_user"))
        session_id = str(
            arguments.pop("session_id", None)
            or request.headers.get("X-Conversation-Id")
            or "mcp_session"
        )

        rag_namespace, course_id = await _resolve_tool_scope(
            container, request, user_id, session_id
        )
        ctx = ToolContext(
            user_id=user_id,
            session_id=session_id,
            rag_namespace=rag_namespace,
            course_id=course_id,
        )
        res = await container.tool_registry.execute(tool_name, arguments, ctx)
        if res.ok:
            try:
                await container.memory_manager.add(
                    type="working",
                    content=f"MCP工具 {tool_name} 执行结果: {res.to_str()[:200]}",
                    user_id=user_id,
                    session_id=session_id,
                    importance=0.6,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("mcp_tool_working_memory_failed", tool=tool_name, error=str(e))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": res.to_str()}],
                "isError": not res.ok,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


# =====================================================================
# 4. llama.cpp WebUI 原生内置工具与断点续传流路由 (/tools, /v1/streams/lookup)
# =====================================================================


@router.get("/tools")
async def llama_webui_list_tools(container: AppContainer = Depends(get_container)):
    """供 llama.cpp WebUI 的 ToolsService.list() 自动加载后端内置工具（RAG、SM-2、记忆等）。"""
    specs = container.tool_registry.specs()
    return [
        {
            "tool": s.name,
            "uses_cwd": s.name == "terminal_tool",
            "definition": {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            },
        }
        for s in specs
    ]


@router.post("/tools")
async def llama_webui_execute_tool(
    request: Request,
    container: AppContainer = Depends(get_container),
):
    """供 llama.cpp WebUI 的 ToolsService.executeTool() / streamTool() 直接执行后端工具。"""
    payload = await request.json()
    tool_name = str(payload.get("tool", ""))
    params = dict(payload.get("params") or {})
    stream = bool(payload.get("stream", False))

    user_id = str(params.pop("user_id", None) or request.headers.get("X-User-Id", "default_user"))
    session_id = str(
        params.pop("session_id", None)
        or request.headers.get("X-Conversation-Id")
        or "webui_tool_session"
    )
    rag_namespace, course_id = await _resolve_tool_scope(container, request, user_id, session_id)
    ctx = ToolContext(
        user_id=user_id,
        session_id=session_id,
        rag_namespace=rag_namespace,
        course_id=course_id,
    )

    res = await container.tool_registry.execute(tool_name, params, ctx)
    if res.ok:
        try:
            await container.memory_manager.add(
                type="working",
                content=f"执行工具 {tool_name} 结果: {res.to_str()[:200]}",
                user_id=user_id,
                session_id=session_id,
                importance=0.6,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("tool_working_memory_failed", tool=tool_name, error=str(e))

    if stream:
        async def _tool_sse() -> AsyncIterator[bytes]:
            if res.ok:
                yield f"data: {json.dumps({'chunk': res.to_str()}, ensure_ascii=False)}\n\n".encode()
                yield b"data: {\"done\": true}\n\n"
            else:
                yield f"data: {json.dumps({'done': True, 'error': res.error or 'Tool execution failed'}, ensure_ascii=False)}\n\n".encode()

        return StreamingResponse(_tool_sse(), media_type="text/event-stream")

    if not res.ok:
        return {"error": res.error or "Tool execution failed"}
    return {"plain_text": res.to_str()}


@router.post("/v1/streams/lookup")
async def llama_streams_lookup(request: Request, container: AppContainer = Depends(get_container)):
    """代理或兜底 llama-server 的 /v1/streams/lookup 断点流查询接口。"""
    return await _proxy_to_llama(request, container, "/v1/streams/lookup", fallback_fn=lambda: [])


@router.get("/v1/stream")
@router.delete("/v1/stream")
async def llama_stream_control(request: Request, container: AppContainer = Depends(get_container)):
    """代理或兜底 llama-server 的 /v1/stream 接口。"""
    return await _proxy_to_llama(request, container, "/v1/stream", fallback_fn=lambda: {"ok": True})


@router.post("/v1/chat/completions/control")
async def llama_chat_control(request: Request, container: AppContainer = Depends(get_container)):
    """代理 llama-server 的实时推理控制接口 (/v1/chat/completions/control)。"""
    return await _proxy_to_llama(
        request,
        container,
        "/v1/chat/completions/control",
        fallback_fn=lambda: {"ok": True},
    )

