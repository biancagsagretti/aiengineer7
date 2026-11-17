from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.schemas.responses import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness: el proceso responde")
async def health():
    return HealthResponse(status="ok", app=get_settings().app_name)


@router.get("/health/ready", response_model=ReadyResponse, summary="Readiness: Qdrant y Redis responden")
async def ready(request: Request):
    settings = get_settings()
    deps: dict[str, str] = {}
    try:
        await request.app.state.vector_store.ping()
        deps["qdrant"] = "ok"
    except Exception as exc:
        deps["qdrant"] = f"error: {type(exc).__name__}"
    try:
        await request.app.state.jobs.ping()
        deps["redis"] = "ok"
    except Exception as exc:
        deps["redis"] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in deps.values())
    body = ReadyResponse(status="ready" if ok else "degraded", provider=settings.normalized_provider, dependencies=deps)
    return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())
