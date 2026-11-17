"""Human-in-the-loop: la ingesta con análisis de imágenes es la acción crítica.

Flujo: ingest_plan (estima, sin gastar tokens) -> approval_gate (interrupt si hay costo) -> ingest_execute | ingest_rejected.
El grafo queda pausado en el checkpoint de Redis hasta que POST /v1/tasks/{id}/approve lo reanuda con Command(resume=...).
"""
from pathlib import Path

from langgraph.types import interrupt

from app.schemas.tasks import IngestPlan


def build_plan(document: str, raw: dict, settings) -> IngestPlan:
    calls = raw["estimated_vision_calls"]
    cost = calls * (
        settings.vision_est_input_tokens * settings.vision_price_in_per_mtok
        + settings.vision_est_output_tokens * settings.vision_price_out_per_mtok
    ) / 1_000_000
    needs = calls >= max(1, settings.hitl_min_vision_calls)
    reason = (
        f"Se harían ~{calls} llamadas al modelo de visión (~USD {cost:.4f} estimado). Requiere aprobación humana."
        if needs else "No hay imágenes pendientes de analizar: no se gastan tokens de visión."
    )
    return IngestPlan(document=document, estimated_cost_usd=round(cost, 4), requires_approval=needs, reason=reason, **raw)


def make_ingest_nodes(ingestion, settings):
    def _path(state) -> Path:
        name = state["document"]
        if name != Path(name).name:
            raise ValueError("Nombre de documento inválido.")
        return Path(settings.documents_dir) / name

    async def ingest_plan(state):
        raw = await ingestion.plan_path(_path(state))
        return {"plan": build_plan(state["document"], raw, settings).model_dump()}

    async def approval_gate(state):
        # interrupt() pausa el grafo y guarda el estado en el checkpointer. Al reanudar, este nodo se
        # vuelve a ejecutar desde el principio y interrupt() devuelve el valor de Command(resume=...).
        decision = interrupt({"action": "vision_ingest", "plan": state["plan"]}) or {}
        plan = state["plan"]
        approved = bool(decision.get("approved", False))
        requested = decision.get("max_vision_calls")
        granted = plan["estimated_vision_calls"] if requested is None else min(int(requested), plan["estimated_vision_calls"])
        return {"approval": {
            "approved": approved,
            "max_vision_calls": granted if approved else 0,
            "reviewer": decision.get("reviewer"),
            "note": decision.get("note"),
        }}

    async def ingest_execute(state):
        plan = state["plan"]
        approval = state.get("approval") or {}
        # Sin aprobación explícita el presupuesto de visión es 0: imposible gastar tokens por accidente.
        budget = approval.get("max_vision_calls", 0) if plan["requires_approval"] else 0
        chunks, warnings = await ingestion.ingest_path(_path(state), vision_budget=budget)
        used = ingestion.vision_calls_used
        return {
            "answer": f"Ingesta completada: {chunks} fragmentos indexados, {used} llamadas de visión.",
            "ingest_result": {"chunks_indexed": chunks, "vision_calls": used, "warnings": warnings},
        }

    async def ingest_rejected(state):
        return {
            "answer": "Ingesta rechazada por el revisor. No se gastaron tokens ni se indexó nada.",
            "ingest_result": {"chunks_indexed": 0, "vision_calls": 0, "warnings": []},
        }

    return {"ingest_plan": ingest_plan, "approval_gate": approval_gate,
            "ingest_execute": ingest_execute, "ingest_rejected": ingest_rejected}
