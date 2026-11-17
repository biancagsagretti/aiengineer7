import hashlib
import uuid
from typing import AsyncIterator

from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings


def point_id(text: str, metadata: dict) -> str:
    """ID determinístico: reingestar el mismo contenido actualiza el punto en vez de duplicarlo."""
    key = "|".join([
        str(metadata.get("document")), str(metadata.get("page")), str(metadata.get("content_type")),
        str(metadata.get("chunk_start", 0)), str(metadata.get("image_path", "")),
        hashlib.sha1(text.encode("utf-8")).hexdigest(),
    ])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._ready = False

    async def ensure_collection(self) -> None:
        if self._ready:
            return
        names = await self.client.get_collections()
        if self.collection not in [c.name for c in names.collections]:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.settings.vector_size, distance=models.Distance.COSINE),
            )
        self._ready = True

    async def upsert(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict]) -> list[str]:
        if not texts:
            return []
        await self.ensure_collection()
        points, ids = [], []
        for text, vector, metadata in zip(texts, vectors, metadatas):
            pid = point_id(text, metadata)
            ids.append(pid)
            points.append(models.PointStruct(id=pid, vector=vector, payload={"text": text, **metadata}))
        await self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return ids

    async def search(self, vector: list[float], limit: int = 6) -> list[dict]:
        await self.ensure_collection()
        # query_points reemplaza a search(), eliminado en versiones recientes de qdrant-client.
        response = await self.client.query_points(
            collection_name=self.collection, query=vector, limit=limit, with_payload=True,
        )
        return [
            {
                "id": str(hit.id),
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "score": hit.score,
            }
            for hit in response.points
        ]

    async def scroll_all(self, batch: int = 256) -> AsyncIterator[dict]:
        """Recorre toda la colección (sin vectores). Sirve para reconstruir el índice BM25 al arrancar."""
        await self.ensure_collection()
        offset = None
        while True:
            points, offset = await self.client.scroll(
                collection_name=self.collection, limit=batch, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for p in points:
                yield {
                    "id": str(p.id),
                    "text": p.payload.get("text", ""),
                    "metadata": {k: v for k, v in p.payload.items() if k != "text"},
                }
            if offset is None:
                break

    async def count(self) -> int:
        await self.ensure_collection()
        return (await self.client.count(collection_name=self.collection, exact=True)).count

    async def ping(self) -> bool:
        await self.client.get_collections()
        return True

    async def close(self) -> None:
        await self.client.close()
