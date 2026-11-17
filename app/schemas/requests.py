from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Level = Literal["principiante", "intermedio", "avanzado", "no_definido"]
Intent = Literal["knowledge", "lesson", "evaluation", "general"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=2, max_length=8000, description="Pregunta del alumno")
    session_id: str = Field(min_length=1, max_length=128, description="Identificador persistente de la sesión")
    level: Level = "principiante"
    intent: Intent | None = Field(default=None, description="Fuerza la intención y omite al Supervisor")


class SourceReference(BaseModel):
    document: str
    page: int | None = None
    content_type: str = "text"
    excerpt: str = ""
    image_url: str | None = Field(default=None, description="URL para ver la imagen original (tablatura, acorde, etc.)")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str
    sources: list[SourceReference] = []
    provider: str
    model: str
    trace_hint: str | None = None


class IngestResponse(BaseModel):
    files_processed: int
    chunks_indexed: int
    warnings: list[str] = []


class IngestJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    filename: str | None = None
    chunks_indexed: int = 0
    warnings: list[str] = []
    error: str | None = None
