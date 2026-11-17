from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from langgraph.types import Command

from app.api.dependencies import require_api_key
from app.schemas.requests import ChatRequest
from app.schemas.tasks import ApprovalRequest, TaskView
from app.worker import chat_graph_input

router = APIRouter(prefix="/v1/tasks", tags=["tasks"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=TaskView, status_code=202, summary="Encola una consulta al agente y devuelve un job_id")
async def create_task(payload: ChatRequest, request: Request, response: Response):
    job = await request.app.state.jobs.create("chat", payload.session_id, payload.model_dump())
    request.app.state.runner.submit(job, chat_graph_input(payload))
    response.headers["Location"] = f"/v1/tasks/{job.job_id}"
    return job


@router.get("", response_model=list[TaskView], summary="Últimos trabajos")
async def list_tasks(request: Request, limit: int = Query(20, ge=1, le=200)):
    return await request.app.state.jobs.list(limit)


@router.get("/{job_id}", response_model=TaskView, summary="Estado de un trabajo (PENDING/RUNNING/WAITING_APPROVAL/DONE/REJECTED/FAILED)")
async def get_task(job_id: str, request: Request):
    job = await request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado o expirado.")
    return job


@router.post("/{job_id}/approve", response_model=TaskView, status_code=202,
             summary="Human-in-the-loop: aprueba o rechaza una ingesta pausada y la reanuda")
async def approve_task(job_id: str, decision: ApprovalRequest, request: Request):
    store = request.app.state.jobs
    # Transición atómica WAITING_APPROVAL -> RUNNING: una segunda aprobación simultánea recibe 409.
    claimed = await store.claim(job_id, "WAITING_APPROVAL", "RUNNING")
    if claimed == -1:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado o expirado.")
    if claimed == 0:
        raise HTTPException(status_code=409, detail="El trabajo no está esperando aprobación.")
    await store.update(job_id, approval=decision.model_dump())
    job = await store.get(job_id)
    request.app.state.runner.submit(job, Command(resume=decision.model_dump()))
    return job
