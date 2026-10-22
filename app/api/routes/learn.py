"""学习领域业务路由（Courses / KnowledgePoints / Quiz / Grade / Review / Progress / Plans）。"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_container, get_user_id
from app.core.container import AppContainer
from app.observability.logging import get_logger
from app.tools.base import ToolContext

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Learning"])


class CourseCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    namespace: str | None = None
    description: str = ""


class KnowledgeProvenance(BaseModel):
    """三方溯源锚点数据结构（对话 ↔ 知识节点 ↔ 教材出处与物理页码）。"""
    message_id: str | None = None
    turn_index: int | None = None
    doc_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    quote_snippet: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class KnowledgeAddRequest(BaseModel):
    name: str = Field(..., min_length=1)
    summary: str = ""
    difficulty: float = Field(0.5, ge=0.1, le=1.0)
    prereq_ids: list[str] = Field(default_factory=list)

    # M2 新增字段（全部具备安全默认值以保持 100% 向后兼容）
    id: str | None = None
    raw_quote: str = ""
    content: str = ""
    provenance: dict[str, Any] | None = None
    user_id: str | None = None
    relations: list[dict[str, Any]] | None = None


class KnowledgeRelationRequest(BaseModel):
    src_id: str = Field(..., min_length=1)
    rel_type: Literal["prerequisite", "causal", "similar", "branch_followup"]
    dst_id: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRollbackRequest(BaseModel):
    target_version: int = Field(..., ge=1)


class QuizGenerateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = "api_quiz"
    topic: str = Field(..., min_length=1)
    question_type: Literal["choice", "fill", "short_answer"] = "choice"
    difficulty: float | None = Field(None, ge=0.1, le=1.0)
    # 可选覆盖：默认跟随会话绑定的 rag_namespace / course_id
    namespace: str | None = None
    course_id: str | None = None


class QuizAnswerRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = "api_quiz"
    user_answer: str = Field(..., min_length=1)
    hints_used: int = Field(0, ge=0)


class StudyPlanCreateRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    deadline: str = ""
    tasks: list[str] = Field(default_factory=list)


@router.post("/courses")
async def create_course(
    req: CourseCreateRequest,
    container: AppContainer = Depends(get_container),
):
    """创建新课程或学科。"""
    course = await container.learning_store.create_course(
        user_id=req.user_id,
        name=req.name,
        namespace=req.namespace,
        description=req.description,
    )
    return {"ok": True, "course": course}


@router.get("/courses/{user_id}")
async def list_courses(
    user_id: str,
    container: AppContainer = Depends(get_container),
):
    """获取指定用户的全部课程列表。"""
    courses = await container.learning_store.list_courses(user_id=user_id)
    return {"user_id": user_id, "count": len(courses), "courses": courses}


@router.post("/courses/{course_id}/knowledge")
async def add_knowledge_point(
    course_id: str,
    req: KnowledgeAddRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """向指定课程新增或更新结构化知识点及前置依赖。"""
    effective_user = (req.user_id or user_id or "default_user").strip() or "default_user"
    try:
        kp = await container.learning_store.upsert_knowledge_point(
            course_id=course_id,
            name=req.name,
            summary=req.summary,
            difficulty=req.difficulty,
            prereq_ids=req.prereq_ids,
            kp_id=req.id,
            raw_quote=req.raw_quote,
            content=req.content,
            provenance=req.provenance,
            user_id=effective_user,
            relations=req.relations,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {
        "status": "ok",
        "ok": True,
        "id": kp["id"],
        "version": kp.get("version", 1),
        "created_at": kp.get("created_at"),
        "user_id": effective_user,
        "course_id": kp.get("course_id", course_id),
        "knowledge_point": kp,
    }


@router.post("/knowledge")
async def add_knowledge_point_alias(
    req: KnowledgeAddRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await add_knowledge_point(
        course_id="default", req=req, user_id=user_id, container=container
    )


@router.get("/courses/{course_id}/knowledge")
async def list_knowledge_points(
    course_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """获取课程下的知识点列表（严格按调用用户过滤）。"""
    cid = None if course_id == "all" else course_id
    kps = await container.learning_store.list_knowledge_points(course_id=cid, user_id=user_id)
    return {
        "course_id": course_id,
        "user_id": user_id,
        "count": len(kps),
        "knowledge_points": kps,
    }


@router.get("/knowledge")
async def list_knowledge_points_alias(
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await list_knowledge_points(
        course_id="default", user_id=user_id, container=container
    )


@router.post("/courses/{course_id}/knowledge/relations", status_code=200)
async def add_knowledge_relation(
    course_id: str,
    req: KnowledgeRelationRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """向指定课程新增知识点间的强类型语义关系。"""
    try:
        rel = await container.learning_store.upsert_relation(
            src_id=req.src_id,
            rel_type=req.rel_type,
            dst_id=req.dst_id,
            metadata=req.metadata,
            user_id=user_id,
            course_id=course_id,
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {
        "status": "ok",
        "ok": True,
        "created": True,
        "relation": rel,
    }


@router.post("/knowledge/relations", status_code=200)
async def add_knowledge_relation_alias(
    req: KnowledgeRelationRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await add_knowledge_relation(
        course_id="default", req=req, user_id=user_id, container=container
    )


@router.get("/courses/{course_id}/knowledge/{kp_id}/relations")
async def get_knowledge_relations(
    course_id: str,
    kp_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """获取指定知识点的全部入边与出边语义关联。"""
    relations = await container.learning_store.get_relations(kp_id=kp_id, user_id=user_id)
    return {
        "status": "ok",
        "ok": True,
        "kp_id": kp_id,
        "count": len(relations),
        "relations": relations,
    }


@router.get("/knowledge/{kp_id}/relations")
async def get_knowledge_relations_alias(
    kp_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await get_knowledge_relations(
        course_id="default", kp_id=kp_id, user_id=user_id, container=container
    )


@router.get("/courses/{course_id}/knowledge/relations")
async def list_course_relations(
    course_id: str,
    rel_type: str | None = Query(None),
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """获取课程下所有语义关系边。"""
    cid = None if course_id == "all" else course_id
    relations = await container.learning_store.list_course_relations(
        course_id=cid, rel_type=rel_type, user_id=user_id
    )
    return {
        "status": "ok",
        "ok": True,
        "course_id": course_id,
        "count": len(relations),
        "relations": relations,
    }


@router.get("/knowledge/relations")
async def list_course_relations_alias(
    rel_type: str | None = Query(None),
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await list_course_relations(
        course_id="all", rel_type=rel_type, user_id=user_id, container=container
    )


@router.delete("/courses/{course_id}/knowledge/relations")
async def delete_knowledge_relation(
    course_id: str,
    src_id: str = Query(...),
    rel_type: str = Query(...),
    dst_id: str = Query(...),
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """删除指定知识点之间的语义连线。"""
    deleted = await container.learning_store.delete_relation(
        src_id_or_id=src_id,
        rel_type=rel_type,
        dst_id=dst_id,
        user_id=user_id,
    )
    return {"status": "ok", "ok": True, "deleted": deleted}


@router.delete("/knowledge/relations")
async def delete_knowledge_relation_alias(
    src_id: str = Query(...),
    rel_type: str = Query(...),
    dst_id: str = Query(...),
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await delete_knowledge_relation(
        course_id="default",
        src_id=src_id,
        rel_type=rel_type,
        dst_id=dst_id,
        user_id=user_id,
        container=container,
    )


@router.delete("/courses/{course_id}/knowledge/{kp_id}")
async def delete_knowledge_point_route(
    course_id: str,
    kp_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """删除知识点并级联清理关系、前置依赖与掌握度记录。"""
    try:
        deleted = await container.learning_store.delete_knowledge_point(kp_id, user_id=user_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Knowledge point {kp_id} not found")
    return {"status": "ok", "ok": True, "deleted": True, "kp_id": kp_id}


@router.delete("/knowledge/{kp_id}")
async def delete_knowledge_point_alias(
    kp_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await delete_knowledge_point_route(
        course_id="default", kp_id=kp_id, user_id=user_id, container=container
    )


@router.post("/courses/{course_id}/knowledge/{kp_id}/rollback")
async def rollback_knowledge_point(
    course_id: str,
    kp_id: str,
    req: KnowledgeRollbackRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """非破坏性回滚知识点内容至指定版本快照。"""
    try:
        kp = await container.learning_store.rollback_knowledge_point(
            kp_id=kp_id,
            target_version=req.target_version,
            user_id=user_id,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "status": "ok",
        "ok": True,
        "id": kp_id,
        "version": kp.get("version"),
        "rolled_back_to": req.target_version,
        "knowledge_point": kp,
    }


@router.post("/knowledge/{kp_id}/rollback")
async def rollback_knowledge_point_alias(
    kp_id: str,
    req: KnowledgeRollbackRequest,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await rollback_knowledge_point(
        course_id="default", kp_id=kp_id, req=req, user_id=user_id, container=container
    )


@router.get("/courses/{course_id}/knowledge/{kp_id}/revisions")
async def get_knowledge_revisions(
    course_id: str,
    kp_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    """获取知识点的历史版本时间轴列表。"""
    kp = await container.learning_store.get_knowledge_point(
        kp_id, course_id=course_id, user_id=user_id
    )
    if not kp:
        raise HTTPException(status_code=404, detail=f"Knowledge point {kp_id} not found")
    revisions = await container.learning_store.get_revisions(kp_id=kp_id, user_id=user_id)
    return {
        "status": "ok",
        "ok": True,
        "kp_id": kp_id,
        "current_version": kp.get("version", 1),
        "count": len(revisions),
        "revisions": revisions,
    }


@router.get("/knowledge/{kp_id}/revisions")
@router.get("/knowledge/{kp_id}/history")
async def get_knowledge_revisions_alias(
    kp_id: str,
    user_id: str = Depends(get_user_id),
    container: AppContainer = Depends(get_container),
):
    return await get_knowledge_revisions(
        course_id="default", kp_id=kp_id, user_id=user_id, container=container
    )


@router.post("/quiz/generate")
async def generate_quiz(
    req: QuizGenerateRequest,
    container: AppContainer = Depends(get_container),
):
    """生成一道结构化测验题并持久化到题库。"""
    quiz_tool = container.tool_registry.get("quiz_tool")
    if not quiz_tool:
        raise HTTPException(status_code=500, detail="quiz_tool not registered")

    # Session binding is authoritative; explicit request fields only override it.
    session = None
    if req.session_id:
        session = await container.state_manager.get_session(
            user_id=req.user_id, session_id=req.session_id
        )
    effective_ns = (
        (req.namespace or "").strip()
        or ((session.rag_namespace or "").strip() if session and session.rag_namespace else "")
        or "default"
    )
    effective_course = (
        (req.course_id or "").strip()
        or ((session.course_id or "").strip() if session and session.course_id else "")
        or None
    )
    ctx = ToolContext(
        user_id=req.user_id,
        session_id=req.session_id,
        rag_namespace=effective_ns,
        course_id=effective_course,
    )
    res = await quiz_tool.run(
        {
            "action": "generate",
            "topic": req.topic,
            "question_type": req.question_type,
            "difficulty": req.difficulty,
        },
        ctx,
    )
    if not res.ok:
        raise HTTPException(status_code=400, detail=res.error or "Failed to generate quiz")
    q = dict(res.metadata)
    q.pop("answer", None)
    q.pop("explanation", None)
    return {"ok": True, "question": q, "display": res.output}


@router.post("/quiz/{question_id}/answer")
async def grade_quiz_answer(
    question_id: str,
    req: QuizAnswerRequest,
    container: AppContainer = Depends(get_container),
):
    """提交题目作答，执行判卷、SM-2 复习周期更新与错题本归档。"""
    grade_tool = container.tool_registry.get("grade_tool")
    if not grade_tool:
        raise HTTPException(status_code=500, detail="grade_tool not registered")

    ctx = ToolContext(user_id=req.user_id, session_id=req.session_id)
    res = await grade_tool.run(
        {
            "question_id": question_id,
            "user_answer": req.user_answer,
            "hints_used": req.hints_used,
        },
        ctx,
    )
    if not res.ok:
        raise HTTPException(status_code=404, detail=res.error or "Failed to grade answer")
    return {"ok": True, "result": res.metadata, "display": res.output}


@router.get("/review/{user_id}/due")
async def get_due_reviews(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    container: AppContainer = Depends(get_container),
):
    """查询用户今日到期需复习的知识点（SM-2）与未解决错题列表。"""
    due_kps = await container.learning_store.get_due_reviews(user_id=user_id, limit=limit)
    mistakes = await container.learning_store.list_mistakes(
        user_id=user_id, resolved=False, limit=limit
    )
    return {
        "user_id": user_id,
        "due_knowledge_points": due_kps,
        "unresolved_mistakes": mistakes,
    }


@router.get("/progress/{user_id}")
async def get_user_progress(
    user_id: str,
    container: AppContainer = Depends(get_container),
):
    """获取用户的全量学情看板（掌握度分布、近期正确率、薄弱知识点、待复习数）。"""
    return await container.learning_store.get_progress_report(user_id=user_id)


@router.post("/plans/{user_id}")
async def create_study_plan(
    user_id: str,
    req: StudyPlanCreateRequest,
    container: AppContainer = Depends(get_container),
):
    """基于用户薄弱点与学习目标生成结构化学习计划。"""
    progress_tool = container.tool_registry.get("progress_tool")
    if not progress_tool:
        raise HTTPException(status_code=500, detail="progress_tool not registered")

    ctx = ToolContext(user_id=user_id, session_id="api_plan")
    res = await progress_tool.run(
        {
            "action": "create_plan",
            "goal": req.goal,
            "deadline": req.deadline,
            "tasks": req.tasks,
        },
        ctx,
    )
    if not res.ok:
        raise HTTPException(status_code=400, detail=res.error or "Failed to create study plan")
    return {"ok": True, "plan": res.metadata, "display": res.output}


@router.get("/plans/{user_id}")
async def list_study_plans(
    user_id: str,
    container: AppContainer = Depends(get_container),
):
    """获取用户的学习计划列表。"""
    plans = await container.learning_store.list_study_plans(user_id=user_id)
    return {"user_id": user_id, "count": len(plans), "plans": plans}
