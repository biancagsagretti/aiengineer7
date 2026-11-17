import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIStatusError, RateLimitError

from app.agents.graph import build_graph, create_checkpointer_cm
from app.api.routes import chat, health, knowledge, media, tasks
from app.core.config import get_settings
from app.core.logging import configure_logging, logger
from app.rag.ingestion import IngestionService, is_daily_quota_error
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from app.services.job_store import JobStore
from app.services.llm_client import LLMClient
from app.services.tracing import initialize_tracing
from app.worker import TaskRunner


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    initialize_tracing(settings)
    llm = LLMClient(settings)
    vector_store = VectorStore(settings)
    retriever = HybridRetriever(vector_store, llm, settings)
    await retriever.warm_up()  # reconstruye BM25 desde Qdrant tras reiniciar
    ingestion = IngestionService(settings, llm, retriever)

    jobs = JobStore(settings.redis_url, settings.job_ttl_seconds)
    stale = await jobs.fail_stale()  # jobs huérfanos de un reinicio anterior => FAILED, no polling infinito
    if stale:
        logger.warning("stale_jobs_marked_failed", count=stale)

    # Checkpoints de LangGraph en Redis (el contexto mantiene abierta la conexión durante la vida de la API).
    async with create_checkpointer_cm(settings) as checkpointer:
        graph = build_graph(llm, retriever, checkpointer, ingestion=ingestion, settings=settings)
        runner = TaskRunner(graph, jobs, settings)
        app.state.settings = settings
        app.state.llm = llm
        app.state.vector_store = vector_store
        app.state.retriever = retriever
        app.state.ingestion = ingestion
        app.state.jobs = jobs
        app.state.runner = runner
        app.state.graph = graph
        logger.info("application_started", provider=settings.normalized_provider, model=settings.active_model,
                    checkpointer=settings.checkpointer_backend)
        try:
            yield
        finally:
            await runner.shutdown()
            await vector_store.close()
            await jobs.close()
            logger.info("application_stopped")


app = FastAPI(
    title="Guitar Intelligence System API",
    version="2.0.0",
    description="Profesor de guitarra con RAG multimodal, Supervisor multiagente, memoria por sesión y trazabilidad (Phoenix).",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Asigna un request_id a cada petición: aparece en logs, en la respuesta y en la metadata del grafo."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RateLimitError)
async def openai_rate_limit(request: Request, exc: RateLimitError):
    daily = is_daily_quota_error(exc)
    detail = ("Se agotó la cuota diaria del proveedor LLM. Probá más tarde o revisá tu plan."
              if daily else "El proveedor LLM está limitando las solicitudes. Reintentá en unos segundos.")
    logger.warning("llm_rate_limited", daily_quota=daily)
    return JSONResponse(status_code=429, content={"detail": detail}, headers={"Retry-After": "30"})


@app.exception_handler(APIConnectionError)
@app.exception_handler(APIStatusError)
async def openai_upstream_error(request: Request, exc: Exception):
    logger.error("llm_upstream_error", error=str(exc)[:300])
    return JSONResponse(status_code=502, content={"detail": "Error al comunicarse con el proveedor LLM."})


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(tasks.router)
app.include_router(knowledge.router)
app.include_router(media.router)
