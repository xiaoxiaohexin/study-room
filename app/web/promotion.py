"""临时知识库「晋升」服务：把提纯后的候选写入标准知识库（文档 + 知识节点）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.learning.store import LearningStore
from app.observability.logging import get_logger
from app.rag.pipeline import IngestionPipeline
from app.storage.sqlite import SQLiteStore
from app.web.staging import WebStagingError, WebStagingService

log = get_logger(__name__)


class PromotionError(Exception):
    """晋升/撤销失败。"""


class PromotionService:
    def __init__(
        self,
        settings: Settings,
        sqlite: SQLiteStore,
        pipeline: IngestionPipeline,
        learning_store: LearningStore,
        staging: WebStagingService,
    ):
        self.settings = settings
        self.sqlite = sqlite
        self.pipeline = pipeline
        self.learning_store = learning_store
        self.staging = staging

    async def _resolve_target(
        self, *, target_course_id: str, target_namespace: str | None, user_id: str
    ) -> tuple[str, str]:
        course_id = (target_course_id or "").strip() or "default"
        namespace = (target_namespace or "").strip()
        if not namespace:
            try:
                course = await self.learning_store.ensure_course(user_id, course_id)
            except PermissionError as exc:
                raise PromotionError(str(exc)) from exc
            namespace = (course.get("namespace") or "default").strip() or "default"
        return course_id, namespace

    async def promote(
        self,
        *,
        staging_doc_ids: list[str],
        candidate_ids: list[str] | None = None,
        target_course_id: str = "default",
        target_namespace: str | None = None,
        user_id: str = "default_user",
    ) -> dict[str, Any]:
        if not staging_doc_ids and not candidate_ids:
            raise PromotionError("请选择要晋升的临时文档或提纯候选")

        course_id, namespace = await self._resolve_target(
            target_course_id=target_course_id,
            target_namespace=target_namespace,
            user_id=user_id,
        )

        promoted_docs: list[dict[str, Any]] = []
        promoted_kps: list[dict[str, Any]] = []
        errors: list[str] = []
        doc_mapping: dict[str, str] = {}

        # 候选未显式提供时，晋升所选文档下的全部 pending 候选
        selected_candidates: list[dict[str, Any]] = []
        if candidate_ids:
            for candidate_id in candidate_ids:
                candidate = await self.sqlite.get_staging_candidate(candidate_id)
                if candidate and candidate.get("status") != "promoted":
                    selected_candidates.append(candidate)
        else:
            for doc_id in staging_doc_ids:
                rows = await self.sqlite.list_staging_candidates(
                    status="pending", doc_id=doc_id, limit=50
                )
                selected_candidates.extend(rows)

        # 需要晋升文档的集合 = 显式选择 + 候选所属文档
        doc_ids = list(dict.fromkeys([*staging_doc_ids, *[c["doc_id"] for c in selected_candidates]]))
        for doc_id in doc_ids:
            try:
                mapping = await self._promote_document(
                    doc_id, namespace=namespace, course_id=course_id, user_id=user_id
                )
                doc_mapping[doc_id] = mapping["doc_id"]
                promoted_docs.append(mapping)
            except (WebStagingError, PromotionError, OSError) as exc:
                errors.append(f"文档 {doc_id} 晋升失败: {exc}")
                log.warning("staging_promotion_doc_failed", doc_id=doc_id, error=str(exc))

        for candidate in selected_candidates:
            new_doc_id = doc_mapping.get(candidate["doc_id"])
            if not new_doc_id:
                errors.append(f"候选 {candidate['id']} 缺少已晋升文档，已跳过")
                continue
            try:
                kp = await self._promote_candidate(
                    candidate,
                    new_doc_id=new_doc_id,
                    course_id=course_id,
                    user_id=user_id,
                )
                promoted_kps.append(kp)
            except (ValueError, PermissionError) as exc:
                errors.append(f"候选 {candidate['id']} 写入知识节点失败: {exc}")
                log.warning("staging_promotion_candidate_failed", candidate=candidate["id"], error=str(exc))

        log.info(
            "staging_promoted",
            documents=len(promoted_docs),
            knowledge_points=len(promoted_kps),
            course_id=course_id,
            namespace=namespace,
        )
        return {
            "ok": not errors,
            "target_course_id": course_id,
            "target_namespace": namespace,
            "promoted_documents": promoted_docs,
            "promoted_knowledge_points": promoted_kps,
            "errors": errors,
        }

    async def _promote_document(
        self, doc_id: str, *, namespace: str, course_id: str, user_id: str
    ) -> dict[str, Any]:
        row = await self.staging.get_staging_document(doc_id)
        if not row:
            raise PromotionError(f"临时文档不存在: {doc_id}")
        if row.get("status") == "promoted" and row.get("promoted_doc_id"):
            existing = await self.sqlite.get_document(row["promoted_doc_id"])
            if existing:
                return {
                    "staging_doc_id": doc_id,
                    "doc_id": existing["doc_id"],
                    "namespace": existing["namespace"],
                    "reused": True,
                }

        copied = self.staging.copy_snapshot(row)
        metadata = {
            "source_type": "web",
            "url": row.get("url") or "",
            "fetched_at": row.get("fetched_at") or 0,
            "content_hash": row.get("content_hash") or "",
            "origin": "staging_promotion",
            "session_id": row.get("session_id") or "",
            "user_id": user_id,
            "metadata_json": json.dumps(
                {
                    "promoted_from_staging_doc": doc_id,
                    "target_course_id": course_id,
                },
                ensure_ascii=False,
            ),
        }
        new_doc_id, chunks = await self.pipeline.ingest_file(
            copied,
            namespace=namespace,
            original_filename=row.get("filename") or "web.md",
            tier="standard",
            metadata=metadata,
        )
        await self.sqlite.update_document(
            doc_id,
            status="promoted",
            promoted_doc_id=new_doc_id,
            target_namespace=namespace,
        )
        log.info(
            "staging_document_promoted",
            staging_doc_id=doc_id,
            doc_id=new_doc_id,
            namespace=namespace,
            chunks=chunks,
        )
        return {
            "staging_doc_id": doc_id,
            "doc_id": new_doc_id,
            "namespace": namespace,
            "chunks": chunks,
            "reused": False,
        }

    async def _promote_candidate(
        self,
        candidate: dict[str, Any],
        *,
        new_doc_id: str,
        course_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        source_doc = await self.sqlite.get_document(candidate["doc_id"])
        provenance = {
            "message_id": "",
            "turn_index": 0,
            "doc_id": new_doc_id,
            "doc_title": (source_doc or {}).get("filename", ""),
            "page_start": None,
            "page_end": None,
            "url": (source_doc or {}).get("url", ""),
            "fetched_at": (source_doc or {}).get("fetched_at", 0),
            "quote_snippet": candidate.get("raw_quote") or "",
            "extra": {
                "origin": "staging_promotion",
                "staging_candidate_id": candidate["id"],
                "staging_doc_id": candidate["doc_id"],
            },
        }
        kp = await self.learning_store.upsert_knowledge_point(
            course_id=course_id,
            name=candidate["title"],
            summary=candidate.get("summary") or "",
            difficulty=float(candidate.get("difficulty") or 0.5),
            raw_quote=candidate.get("raw_quote") or "",
            content=candidate.get("content") or candidate.get("summary") or "",
            provenance=provenance,
            user_id=user_id,
        )
        await self.sqlite.update_staging_candidate(
            candidate["id"],
            status="promoted",
            promoted_kp_id=kp["id"],
            promoted_doc_id=new_doc_id,
            updated_at=time.time(),
        )
        return {
            "candidate_id": candidate["id"],
            "kp_id": kp["id"],
            "name": kp["name"],
            "source_doc_id": new_doc_id,
        }

    async def revoke(self, staging_doc_id: str, *, user_id: str = "default_user") -> dict[str, Any]:
        row = await self.staging.get_staging_document(staging_doc_id)
        if not row:
            raise PromotionError(f"临时文档不存在: {staging_doc_id}")
        if row.get("status") != "promoted":
            raise PromotionError("该临时文档尚未晋升，无需撤销")

        removed_doc = ""
        promoted_doc_id = row.get("promoted_doc_id") or ""
        if promoted_doc_id:
            promoted = await self.sqlite.get_document(promoted_doc_id)
            if promoted:
                namespace = promoted.get("namespace") or "default"
                collection = self.pipeline._collection_name(namespace)
                chunk_count = int(promoted.get("chunk_count") or 0)
                ids = [f"{promoted_doc_id}_{i}" for i in range(chunk_count)]
                if ids:
                    await self.pipeline.vector_store.delete(collection, ids)
                file_path = promoted.get("file_path") or ""
                data_dir = Path(self.settings.data_dir).resolve()
                if file_path:
                    candidate_path = Path(file_path)
                    try:
                        if candidate_path.resolve().is_relative_to(data_dir):
                            candidate_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                await self.sqlite.delete_document(promoted_doc_id)
                removed_doc = promoted_doc_id

        removed_kps: list[str] = []
        candidates = await self.sqlite.list_staging_candidates(
            status="promoted", doc_id=staging_doc_id, limit=100
        )
        for candidate in candidates:
            kp_id = candidate.get("promoted_kp_id") or ""
            if kp_id:
                try:
                    if await self.learning_store.delete_knowledge_point(kp_id, user_id=user_id):
                        removed_kps.append(kp_id)
                except PermissionError:
                    pass
            await self.sqlite.update_staging_candidate(
                candidate["id"],
                status="pending",
                promoted_kp_id="",
                promoted_doc_id="",
                updated_at=time.time(),
            )

        await self.sqlite.update_document(
            staging_doc_id, status="distilled", promoted_doc_id=""
        )
        log.info(
            "staging_promotion_revoked",
            staging_doc_id=staging_doc_id,
            removed_doc=removed_doc,
            revoked_knowledge_points=len(removed_kps),
        )
        return {
            "ok": True,
            "staging_doc_id": staging_doc_id,
            "removed_doc_id": removed_doc,
            "revoked_knowledge_points": removed_kps,
        }
