from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from app.api.dependencies import require_api_key

router = APIRouter(prefix="/v1/media", tags=["media"], dependencies=[Depends(require_api_key)])


@router.get("/{filename}", summary="Imagen original (tablatura, acorde, partitura) citada en una fuente")
async def get_image(filename: str, request: Request):
    base = (Path(request.app.state.settings.processed_dir) / "images").resolve()
    target = (base / filename).resolve()
    if target.parent != base or not target.is_file():  # bloquea ../ y rutas absolutas
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    return FileResponse(target)
