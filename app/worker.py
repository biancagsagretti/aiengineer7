"""Worker en segundo plano: ejecuta el grafo, mantiene el estado del job en Redis y captura TODA excepción (-> FAILED)."""
import asyncio
from typing import Any

from langgraph.types import Command

from app.core.logging import logger
from app.schemas.tasks import TaskView
from app.services.job_store import JobStore
from app.services.tracing import span


def chat_graph_input(payload) -> dict[str, Any]:
    return {
        "question": payload.question,
        "session_id": payload.session_id,
        "level": payload.level,
        "forced_intent": payload.intent or "",
        "document": "",
    }


def ingest_graph_input(job_id: str, document: str) -> dict[str, Any]:
    return {
        "question": f"Ingestar {document}", "session_id": f"ingest-{job_id}", "level": "no_definido",
        "forced_intent": "", "document": document,
    }


class TaskRunner:
    def __init__(self, graph, store: JobStore, settings):
        self.graph, self.store, self.settings = graph, store, settings
        self._sem = asyncio.Semaphore(settings.worker_concurrency)
        self._tasks: set[asyncio.Task] = set()

    def submit(self, job: TaskView, payload: dict[str, Any] | Command) -> None:
        task = asyncio.create_task(self._run(job, payload))
        self._tasks.add(task)  # referencia fuerte: si no, el GC puede destruir la tarea
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, job: TaskView, payload: dict[str, Any] | Command) -> None:
        config = {"configurable": {"thread_id": job.thread_id}, "metadata": {"job_id": job.job_id, "kind": job.kind}}
        async with self._sem:  # límite de concurrencia; NO se retiene mientras se espera la aprobación humana
            try:
                await self.store.update(job.job_id, status="RUNNING")
                # Un span raíz por ejecución: en Phoenix cada tarea es UNA traza (costo y latencia por ejecución).
                with span("task.run", new_root=True, **{"job.id": job.job_id, "job.kind": job.kind, "session.id": job.thread_id,
                                         "openinference.span.kind": "CHAIN"}):
                    await self.graph.ainvoke(payload, config=config)
                    snapshot = await self.graph.aget_state(config)
                await self._finish(job, snapshot)
            except asyncio.CancelledError:
                await self.store.update(job.job_id, status="FAILED", error="Tarea cancelada (apagado del servicio).")
                raise
            except Exception as exc:  # nunca dejar al cliente haciendo polling de un job que ya murió
                logger.exception("task_failed", job_id=job.job_id, error=str(exc))
                try:
                    await self.store.update(job.job_id, status="FAILED", error=f"{type(exc).__name__}: {str(exc)[:500]}")
                except Exception:
                    logger.exception("task_failed_state_update_failed", job_id=job.job_id)

    async def _finish(self, job: TaskView, snapshot) -> None:
        values = snapshot.values
        if snapshot.next:  # el grafo se pausó en approval_gate
            await self.store.update(job.job_id, status="WAITING_APPROVAL", plan=values.get("plan"))
            logger.info("task_waiting_approval", job_id=job.job_id)
            return
        if job.kind == "ingest":
            approval = values.get("approval") or {}
            rejected = values.get("plan", {}).get("requires_approval") and not approval.get("approved")
            await self.store.update(
                job.job_id, status="REJECTED" if rejected else "DONE",
                plan=values.get("plan"), approval=values.get("approval"),
                result={"answer": values.get("answer"), **(values.get("ingest_result") or {})},
            )
            return
        await self.store.update(job.job_id, status="DONE", result={
            "answer": values.get("answer"), "intent": values.get("intent"), "sources": values.get("sources", []),
            "provider": self.settings.normalized_provider, "model": self.settings.active_model,
        })
