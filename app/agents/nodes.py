import json
import os

from app.agents.prompts import SUPERVISOR_PROMPT, RAG_PROMPT, TUTOR_PROMPT, PLANNER_PROMPT, EVALUATOR_PROMPT
from app.core.logging import logger
from app.schemas.music import SupervisorDecision

MEDIA_PREFIX = "/v1/media/"
HISTORY_TURNS = 6  # mensajes (no turnos) que se envían al modelo


def _history_text(state) -> str:
    items = state.get("history", [])[-HISTORY_TURNS:]
    if not items:
        return ""
    lines = [f"{m['role']}: {m['content'][:500]}" for m in items]
    return "Historial reciente:\n" + "\n".join(lines) + "\n"


def _context_text(state) -> str:
    # Se omiten rutas y scores: son ruido para el modelo y gastan tokens.
    slim = [
        {k: c.get(k) for k in ("text", "document", "page", "content_type")}
        for c in state.get("context", [])
    ]
    return json.dumps(slim, ensure_ascii=False)


def _image_url(image_path: str | None) -> str | None:
    return f"{MEDIA_PREFIX}{os.path.basename(image_path)}" if image_path else None


def make_nodes(llm, retriever):
    async def supervisor(state):
        if state.get("document"):  # tarea de ingesta: no hace falta gastar una llamada al LLM
            return {"intent": "ingest", "reasoning": "tarea de ingesta de documento", "context": [], "sources": []}
        forced = state.get("forced_intent") or ""
        if forced:
            return {"intent": forced, "reasoning": "intención indicada por el cliente", "context": [], "sources": []}
        prompt = f"{_history_text(state)}Mensaje del alumno: {state['question']}"
        decision = await llm.generate_structured(SUPERVISOR_PROMPT, prompt, SupervisorDecision)
        if decision is None:
            decision = SupervisorDecision(intent="knowledge", reasoning="fallback: salida del supervisor inválida")
        logger.info("supervisor_decision", intent=decision.intent, reasoning=decision.reasoning)
        return {"intent": decision.intent, "reasoning": decision.reasoning, "context": [], "sources": []}

    async def retrieve(state):
        query = state["question"]
        prev = [m["content"] for m in state.get("history", []) if m["role"] == "user"]
        if prev and len(query.split()) < 8:
            query = f"{prev[-1]} {query}"  # repregunta corta: hereda el tema anterior
        results = await retriever.retrieve(query)
        context, sources = [], []
        for item in results:
            meta = item.get("metadata", {})
            base = {
                "document": meta.get("document", "desconocido"),
                "page": meta.get("page"),
                "content_type": meta.get("music_type") or meta.get("content_type", "text"),
            }
            context.append({**base, "text": item.get("text", "")})
            sources.append({**base, "excerpt": item.get("text", "")[:300], "image_url": _image_url(meta.get("image_path"))})
        return {"context": context, "sources": sources}

    async def rag_agent(state):
        prompt = f"{_history_text(state)}Pregunta: {state['question']}\nNivel: {state['level']}\nContexto: {_context_text(state)}"
        return {"answer": await llm.generate(RAG_PROMPT, prompt)}

    async def tutor(state):
        prompt = (
            f"{_history_text(state)}Pregunta: {state['question']}\nNivel: {state['level']}\n"
            f"Contexto recuperado: {_context_text(state)}\nBorrador fundamentado: {state.get('answer', '')}"
        )
        return {"answer": await llm.generate(TUTOR_PROMPT, prompt)}

    async def planner(state):
        prompt = f"{_history_text(state)}Solicitud: {state['question']}\nNivel: {state['level']}\nContexto disponible: {_context_text(state)}"
        return {"answer": await llm.generate(PLANNER_PROMPT, prompt)}

    async def evaluator(state):
        prompt = f"{_history_text(state)}Solicitud: {state['question']}\nNivel: {state['level']}\nContexto: {_context_text(state)}"
        return {"answer": await llm.generate(EVALUATOR_PROMPT, prompt)}

    async def general(state):
        note = ""
        if state.get("intent") != "general":
            note = ("AVISO: no se encontró material relacionado en los documentos cargados. "
                    "Aclará al alumno que esta respuesta es general y no proviene de sus documentos.\n")
        prompt = f"{note}{_history_text(state)}Nivel: {state['level']}\nPregunta: {state['question']}"
        return {"context": [], "sources": [], "answer": await llm.generate(TUTOR_PROMPT, prompt)}

    async def remember(state):
        return {"history": [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": state.get("answer", "")},
        ]}

    return {
        "supervisor": supervisor, "retrieve": retrieve, "rag_agent": rag_agent, "tutor": tutor,
        "planner": planner, "evaluator": evaluator, "general": general, "remember": remember,
    }
