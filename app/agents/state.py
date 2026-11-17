import operator
from typing import Annotated, Any, NotRequired, TypedDict


class GuitarState(TypedDict):
    question: str
    session_id: str
    level: str
    forced_intent: NotRequired[str]  # "" = que decida el Supervisor
    intent: NotRequired[str]
    reasoning: NotRequired[str]
    context: NotRequired[list[dict[str, Any]]]
    answer: NotRequired[str]
    sources: NotRequired[list[dict[str, Any]]]
    # Ingesta con aprobación humana (HITL)
    document: NotRequired[str]            # nombre del archivo en DOCUMENTS_DIR; "" = tarea de chat
    plan: NotRequired[dict[str, Any]]
    approval: NotRequired[dict[str, Any]]
    ingest_result: NotRequired[dict[str, Any]]
    # Memoria de la conversación: el checkpointer (Redis) la persiste por thread_id.
    history: Annotated[list[dict[str, str]], operator.add]
