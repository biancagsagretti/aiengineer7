from fastapi import Header, HTTPException, Request
from app.core.config import get_settings


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)):
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="API key inválida o ausente.")
