import asyncio

import pytest

from app.rag.chunking import split_text
from app.rag.retriever import reciprocal_rank_fusion, tokenize
from app.rag.vector_store import point_id
from app.schemas.music import MusicInterpretation, SupervisorDecision
from app.schemas.requests import ChatRequest


def test_chunking_overlap_and_validation():
    chunks = split_text("palabra " * 500, {"document": "a.pdf"}, chunk_size=200, overlap=40)
    assert len(chunks) > 1 and all(len(c.text) <= 200 for c in chunks)
    with pytest.raises(ValueError):
        split_text("x", {}, chunk_size=10, overlap=10)


def test_point_id_is_deterministic():
    meta = {"document": "a.pdf", "page": 1, "content_type": "text", "chunk_start": 0}
    assert point_id("hola", meta) == point_id("hola", meta)
    assert point_id("hola", meta) != point_id("chau", meta)


def test_rrf_prefers_items_in_both_rankings():
    a = {"id": "1", "text": "a", "metadata": {}}
    b = {"id": "2", "text": "b", "metadata": {}}
    c = {"id": "3", "text": "c", "metadata": {}}
    fused = reciprocal_rank_fusion([a, b], [c, b])
    assert fused[0]["id"] == "2"


def test_tokenize_keeps_chord_names():
    assert "am7" in tokenize("Acorde Am7, trasto 5")


def test_chat_request_validation():
    with pytest.raises(Exception):
        ChatRequest(question="hola", session_id="s", level="experto")
    assert ChatRequest(question="  hola  ", session_id="s").question == "hola"


def test_music_interpretation_bounds_and_text():
    with pytest.raises(Exception):
        MusicInterpretation(description="x", confidence=1.5)
    m = MusicInterpretation(content_type="chord_diagram", description="Am", extracted_symbols=["x02210"])
    assert "x02210" in m.as_index_text()
    assert SupervisorDecision(intent="lesson").intent == "lesson"


class FakeLLM:
    async def generate_structured(self, system, user, schema):
        return schema(intent="knowledge", reasoning="test")

    async def generate(self, system, user, **kw):
        return "respuesta"


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    async def retrieve(self, query):
        return self.results


def _run_graph(results, question="¿Qué es un acorde de Am?"):
    from langgraph.checkpoint.memory import MemorySaver
    from app.agents.graph import build_graph

    graph = build_graph(FakeLLM(), FakeRetriever(results), MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    inp = {"question": question, "session_id": "t1", "level": "principiante", "forced_intent": ""}
    return asyncio.run(graph.ainvoke(inp, config=cfg)), graph, cfg


def test_graph_with_sources_and_memory():
    hit = {"text": "Am = x02210", "metadata": {"document": "a.pdf", "page": 3, "image_path": "/d/x.png"}}
    result, graph, cfg = _run_graph([hit])
    assert result["intent"] == "knowledge" and result["sources"][0]["image_url"] == "/v1/media/x.png"
    assert len(result["history"]) == 2
    again = asyncio.run(graph.ainvoke({"question": "¿y el de Em?", "session_id": "t1", "level": "principiante", "forced_intent": ""}, config=cfg))
    assert len(again["history"]) == 4  # el checkpointer conserva la conversación


def test_graph_without_context_falls_back_to_general():
    result, _, _ = _run_graph([])
    assert result["sources"] == [] and result["answer"]
