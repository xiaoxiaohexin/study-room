"""学习领域业务路由（Courses / KnowledgePoints / Quiz / Grade / Review / Progress / Plans）。"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_container
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


class KnowledgeAddRequest(BaseModel):
    name: str = Field(..., min_length=1)
    summary: str = ""
    difficulty: float = Field(0.5, ge=0.1, le=1.0)
    prereq_ids: list[str] = Field(default_factory=list)


class QuizGenerateRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = "api_quiz"
    topic: str = Field(..., min_length=1)
    question_type: Literal["choice", "fill", "short_answer"] = "choice"
    difficulty: float | None = Field(None, ge=0.1, le=1.0)
    namespace: str = "default"


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
    container: AppContainer = Depends(get_container),
):
    """向指定课程新增或更新结构化知识点及前置依赖。"""
    kp = await container.learning_store.upsert_knowledge_point(
        course_id=course_id,
        name=req.name,
        summary=req.summary,
        difficulty=req.difficulty,
        prereq_ids=req.prereq_ids,
    )
    return {"ok": True, "knowledge_point": kp}


@router.get("/courses/{course_id}/knowledge")
async def list_knowledge_points(
    course_id: str,
    container: AppContainer = Depends(get_container),
):
    """获取课程下的知识点列表。"""
    cid = None if course_id == "all" else course_id
    kps = await container.learning_store.list_knowledge_points(course_id=cid)
    return {"course_id": course_id, "count": len(kps), "knowledge_points": kps}


@router.post("/quiz/generate")
async def generate_quiz(
    req: QuizGenerateRequest,
    container: AppContainer = Depends(get_container),
):
    """生成一道结构化测验题并持久化到题库。"""
    quiz_tool = container.tool_registry.get("quiz_tool")
    if not quiz_tool:
        raise HTTPException(status_code=500, detail="quiz_tool not registered")

    ctx = ToolContext(user_id=req.user_id, session_id=req.session_id)
    res = await quiz_tool.run(
        {
            "action": "generate",
            "topic": req.topic,
            "question_type": req.question_type,
            "difficulty": req.difficulty,
            "namespace": req.namespace,
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
