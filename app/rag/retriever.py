import re

from rank_bm25 import BM25Okapi

from app.core.logging import logger
from app.rag.chunking import split_text
from app.services.tracing import span

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(*rankings: list[dict], k: int = 60) -> list[dict]:
    """Fusiona listas ordenadas sin comparar escalas de score distintas."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            key = item.get("id") or f"{item['metadata'].get('document')}|{item['metadata'].get('page')}|{item['text'][:80]}"
            entry = fused.setdefault(key, {**item, "rrf": 0.0})
            entry["rrf"] += 1 / (k + rank)
    return sorted(fused.values(), key=lambda item: item["rrf"], reverse=True)


class HybridRetriever:
    """Vectorial (Qdrant) + léxico (BM25 en memoria) fusionados con RRF."""

    def __init__(self, vector_store, llm, settings):
        self.vector_store = vector_store
        self.llm = llm
        self.settings = settings
        self._lexical: dict[str, dict] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_order: list[str] = []

    # ---- índice léxico -------------------------------------------------
    def _add_lexical(self, pid: str, text: str, metadata: dict) -> None:
        self._lexical[pid] = {"id": pid, "text": text, "metadata": metadata}
        self._bm25 = None  # se reconstruye perezosamente en la próxima búsqueda

    async def warm_up(self) -> int:
        """Reconstruye BM25 desde Qdrant: sin esto, tras reiniciar la API la búsqueda léxica queda vacía."""
        try:
            async for item in self.vector_store.scroll_all():
                self._add_lexical(item["id"], item["text"], item["metadata"])
        except Exception as exc:
            logger.warning("lexical_warmup_failed", error=str(exc))
        logger.info("lexical_index_ready", documents=len(self._lexical))
        return len(self._lexical)

    def _ensure_bm25(self) -> None:
        if self._bm25 is None and self._lexical:
            self._bm25_order = list(self._lexical)
            self._bm25 = BM25Okapi([tokenize(self._lexical[i]["text"]) or ["_"] for i in self._bm25_order])

    # ---- indexación ----------------------------------------------------
    async def index_text(self, text: str, metadata: dict) -> int:
        chunks = split_text(text, metadata, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectors = await self.llm.embed_texts(texts)
        ids = await self.vector_store.upsert(texts, vectors, metadatas)
        for pid, t, m in zip(ids, texts, metadatas):
            self._add_lexical(pid, t, m)
        return len(chunks)

    # ---- recuperación --------------------------------------------------
    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.settings.top_k
        with span("rag.retrieve", **{"rag.query": query[:200], "rag.top_k": k}) as current:
            query_vector = (await self.llm.embed_texts([query]))[0]
            vector_results = await self.vector_store.search(query_vector, limit=k * 2)
            lexical_results: list[dict] = []
            self._ensure_bm25()
            if self._bm25 is not None:
                scores = self._bm25.get_scores(tokenize(query))
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: k * 2]
                lexical_results = [
                    {**self._lexical[self._bm25_order[i]], "score": float(s)} for i, s in ranked if s > 0
                ]
            results = reciprocal_rank_fusion(vector_results, lexical_results)[:k]
            if current is not None:
                current.set_attribute("rag.vector_hits", len(vector_results))
                current.set_attribute("rag.lexical_hits", len(lexical_results))
                current.set_attribute("rag.returned", len(results))
            return results
