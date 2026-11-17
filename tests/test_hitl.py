import asyncio
from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agents.graph import build_graph
from app.hitl import build_plan
from app.worker import ingest_graph_input

SETTINGS = SimpleNamespace(
    documents_dir="/tmp/docs", hitl_min_vision_calls=1,
    vision_est_input_tokens=1800, vision_est_output_tokens=350,
    vision_price_in_per_mtok=0.40, vision_price_out_per_mtok=1.60,
)


class FakeIngestion:
    def __init__(self, calls):
        self.calls, self.executed, self._used = calls, [], 0

    async def plan_path(self, path):
        return {"text_records_pending": 3, "images_total": 10, "images_small_skipped": 0,
                "images_pending": self.calls, "estimated_vision_calls": self.calls}

    async def ingest_path(self, path, vision_budget=None):
        self.executed.append(vision_budget)
        self._used = vision_budget or 0
        return 5, []

    @property
    def vision_calls_used(self):
        return self._used


class _LLM:  # no se usa en el camino de ingesta
    pass


class _Retriever:
    pass


def _setup(calls):
    ing = FakeIngestion(calls)
    graph = build_graph(_LLM(), _Retriever(), MemorySaver(), ingestion=ing, settings=SETTINGS)
    cfg = {"configurable": {"thread_id": "ingest-j1"}}
    asyncio.run(graph.ainvoke(ingest_graph_input("j1", "doc.pdf"), config=cfg))
    return graph, cfg, ing


def test_build_plan_estimates_cost_and_flags_approval():
    plan = build_plan("a.pdf", {"estimated_vision_calls": 100, "images_pending": 100, "images_total": 120,
                                "images_small_skipped": 5, "text_records_pending": 10}, SETTINGS)
    assert plan.requires_approval and abs(plan.estimated_cost_usd - 0.128) < 1e-6
    free = build_plan("a.pdf", {"estimated_vision_calls": 0}, SETTINGS)
    assert not free.requires_approval and free.estimated_cost_usd == 0


def test_graph_pauses_and_spends_nothing_before_approval():
    graph, cfg, ing = _setup(8)
    snap = asyncio.run(graph.aget_state(cfg))
    assert snap.next == ("approval_gate",)
    assert snap.values["plan"]["requires_approval"] is True
    assert ing.executed == []  # ninguna llamada de visión antes de aprobar


def test_approval_resumes_with_capped_budget():
    graph, cfg, ing = _setup(8)
    asyncio.run(graph.ainvoke(Command(resume={"approved": True, "max_vision_calls": 3}), config=cfg))
    assert ing.executed == [3]


def test_approval_cannot_exceed_estimate():
    graph, cfg, ing = _setup(8)
    asyncio.run(graph.ainvoke(Command(resume={"approved": True, "max_vision_calls": 999}), config=cfg))
    assert ing.executed == [8]


def test_rejection_executes_nothing():
    graph, cfg, ing = _setup(8)
    result = asyncio.run(graph.ainvoke(Command(resume={"approved": False}), config=cfg))
    assert ing.executed == [] and "rechazada" in result["answer"].lower()


def test_no_pending_images_skips_approval_and_forbids_vision():
    graph, cfg, ing = _setup(0)
    snap = asyncio.run(graph.aget_state(cfg))
    assert snap.next == () and ing.executed == [0]  # presupuesto de visión 0
