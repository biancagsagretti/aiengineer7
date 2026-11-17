from langgraph.graph import StateGraph, START, END

from app.agents.nodes import make_nodes
from app.agents.state import GuitarState
from app.hitl import make_ingest_nodes


def create_checkpointer_cm(settings):
    """Devuelve un async context manager con el checkpointer (Redis por defecto)."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _cm():
        if settings.checkpointer_backend == "memory":
            from langgraph.checkpoint.memory import MemorySaver
            yield MemorySaver()
            return
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        async with AsyncRedisSaver.from_conn_string(settings.redis_url) as saver:
            await saver.asetup()
            yield saver

    return _cm()


def build_graph(llm, retriever, checkpointer, ingestion=None, settings=None):
    """Supervisor -> chat (retrieve -> rag_agent -> tutor|planner|evaluator -> remember)
                  -> ingesta (ingest_plan -> [approval_gate: INTERRUPT] -> ingest_execute | ingest_rejected)."""
    nodes = make_nodes(llm, retriever)
    if ingestion is not None:
        nodes.update(make_ingest_nodes(ingestion, settings))
    graph = StateGraph(GuitarState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    def after_supervisor(s):
        if s.get("document"):
            return "ingest_plan"
        return "general" if s.get("intent") == "general" else "retrieve"

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", after_supervisor,
        {"ingest_plan": "ingest_plan", "general": "general", "retrieve": "retrieve"} if ingestion is not None
        else {"general": "general", "retrieve": "retrieve"},
    )
    graph.add_conditional_edges(
        "retrieve", lambda s: "rag_agent" if s.get("context") else "general",
        {"rag_agent": "rag_agent", "general": "general"},
    )
    graph.add_conditional_edges(
        "rag_agent",
        lambda s: {"lesson": "planner", "evaluation": "evaluator"}.get(s.get("intent"), "tutor"),
        {"planner": "planner", "evaluator": "evaluator", "tutor": "tutor"},
    )
    for terminal in ("planner", "evaluator", "tutor", "general"):
        graph.add_edge(terminal, "remember")
    graph.add_edge("remember", END)

    if ingestion is not None:
        graph.add_conditional_edges(
            "ingest_plan",
            lambda s: "approval_gate" if s["plan"]["requires_approval"] else "ingest_execute",
            {"approval_gate": "approval_gate", "ingest_execute": "ingest_execute"},
        )
        graph.add_conditional_edges(
            "approval_gate",
            lambda s: "ingest_execute" if s["approval"]["approved"] else "ingest_rejected",
            {"ingest_execute": "ingest_execute", "ingest_rejected": "ingest_rejected"},
        )
        graph.add_edge("ingest_execute", END)
        graph.add_edge("ingest_rejected", END)
    return graph.compile(checkpointer=checkpointer)
