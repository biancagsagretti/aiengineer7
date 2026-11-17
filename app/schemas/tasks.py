from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["PENDING", "RUNNING", "WAITING_APPROVAL", "DONE", "REJECTED", "FAILED"]
JobKind = Literal["chat", "ingest"]


class IngestPlan(BaseModel):
    """Lo que el agente estima ANTES de gastar tokens; es lo que revisa la persona que aprueba."""

    document: str
    text_records_pending: int = 0
    images_total: int = 0
    images_pending: int = 0
    images_small_skipped: int = 0
    estimated_vision_calls: int = 0
    estimated_cost_usd: float = 0.0
    requires_approval: bool = False
    reason: str = ""


class ApprovalRequest(BaseModel):
    approved: bool = Field(description="true = autoriza el gasto de visión; false = rechaza la tarea")
    max_vision_calls: int | None = Field(default=None, ge=0, description="Tope de llamadas autorizadas (por defecto, las estimadas)")
    reviewer: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=500)


class TaskView(BaseModel):
    job_id: str
    kind: JobKind
    status: JobStatus
    thread_id: str
    created_at: float
    updated_at: float
    request: dict[str, Any] = {}
    plan: IngestPlan | None = None
    approval: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
