import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import require_api_key
from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.requests import ChatRequest, ChatResponse, SourceReference
from app.worker import chat_graph_input

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(require_api_key)])


def _config(payload: ChatRequest, request_id: str) -> dict[str, Any]:
    # thread_id => el checkpointer de Redis recupera la memoria de la sesión.
    return {"configurable": {"thread_id": payload.session_id}, "metadata": {"request_id": request_id}}


def _build_response(payload: ChatRequest, result: dict, request_id: str) -> ChatResponse:
    settings = get_settings()
    return ChatResponse(
        session_id=payload.session_id,
        answer=result.get("answer") or "No pude generar una respuesta.",
        intent=result.get("intent", "knowledge"),
        sources=[SourceReference(**s) for s in result.get("sources", [])],
        provider=settings.normalized_provider,
        model=settings.active_model,
        trace_hint=request_id,
    )


@router.post("/chat", response_model=ChatResponse, summary="Pregunta al profesor de guitarra")
async def chat(payload: ChatRequest, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info("chat_request_started", session_id=payload.session_id, level=payload.level)
    result = await request.app.state.graph.ainvoke(chat_graph_input(payload), config=_config(payload, request_id))
    logger.info("chat_request_completed", session_id=payload.session_id, intent=result.get("intent"))
    return _build_response(payload, result, request_id)


@router.post("/chat/stream", summary="Igual que /chat pero emite cada paso del grafo por SSE")
async def chat_stream(payload: ChatRequest, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    graph = request.app.state.graph

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def events():
        merged: dict[str, Any] = {}
        try:
            async for update in graph.astream(chat_graph_input(payload), config=_config(payload, request_id), stream_mode="updates"):
                for node, delta in update.items():
                    delta = delta or {}
                    merged.update({k: v for k, v in delta.items() if k != "history"})
                    step: dict[str, Any] = {"node": node}
                    if node == "supervisor":
                        step.update(intent=delta.get("intent"), reasoning=delta.get("reasoning"))
                    elif node == "retrieve":
                        step["sources_found"] = len(delta.get("sources", []))
                    yield sse("step", step)
            yield sse("answer", _build_response(payload, merged, request_id).model_dump())
        except Exception as exc:  # el cliente SSE ya recibió 200: el error viaja como evento
            logger.exception("chat_stream_failed", error=str(exc))
            yield sse("error", {"detail": "Error procesando la solicitud.", "type": type(exc).__name__})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Request-ID": request_id, "Cache-Control": "no-cache"})
