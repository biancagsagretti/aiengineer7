import hashlib
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile

from app.api.dependencies import require_api_key
from app.schemas.tasks import TaskView
from app.worker import ingest_graph_input

router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"], dependencies=[Depends(require_api_key)])
ALLOWED = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


async def _enqueue(request: Request, document: str) -> TaskView:
    job = await request.app.state.jobs.create("ingest", None, {"document": document})
    request.app.state.runner.submit(job, ingest_graph_input(job.job_id, document))
    return job


@router.post("/ingest", response_model=TaskView, status_code=202,
             summary="Sube un PDF/DOCX. El agente estima el costo de visión y pide aprobación antes de gastar tokens")
async def ingest_file(request: Request, response: Response, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=415, detail="Formato no soportado. Subí un PDF o DOCX.")
    target_dir = Path(request.app.state.settings.documents_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Nombre con el hash del contenido: re-subir el mismo archivo reutiliza el estado de la ingesta.
    tmp = target_dir / f".upload_{uuid.uuid4().hex}"
    digest, size = hashlib.sha256(), 0
    with tmp.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx. 200 MB).")
            digest.update(chunk)
            out.write(chunk)
    clean = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "document").name)
    target = target_dir / f"{digest.hexdigest()[:12]}_{clean}"
    tmp.replace(target)

    job = await _enqueue(request, target.name)
    response.headers["Location"] = f"/v1/tasks/{job.job_id}"
    return job


@router.post("/ingest-folder", response_model=list[TaskView], status_code=202,
             summary="Crea una tarea de ingesta por cada PDF/DOCX de DOCUMENTS_DIR")
async def ingest_folder(request: Request):
    folder = Path(request.app.state.settings.documents_dir)
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED)
    return [await _enqueue(request, p.name) for p in files]


@router.get("/stats", summary="Cuántos fragmentos hay indexados")
async def stats(request: Request):
    return {
        "indexed_points": await request.app.state.vector_store.count(),
        "lexical_documents": len(request.app.state.retriever._lexical),
    }
