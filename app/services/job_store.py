import json
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from app.schemas.tasks import TaskView

# Compare-and-set atómico sobre el campo "status": evita que dos aprobaciones simultáneas reanuden el mismo job.
_CAS = """
if redis.call('EXISTS', KEYS[1]) == 0 then return -1 end
if redis.call('HGET', KEYS[1], 'status') == ARGV[1] then
  redis.call('HSET', KEYS[1], 'status', ARGV[2], 'updated_at', ARGV[3])
  return 1
end
return 0
"""


class JobStore:
    """Estado de los jobs en Redis (hash por job + índice ordenado por fecha, con TTL)."""

    INDEX = "jobs:index"

    def __init__(self, url: str, ttl_seconds: int):
        self.redis = aioredis.from_url(url, decode_responses=True)
        self.ttl = ttl_seconds
        self._cas = self.redis.register_script(_CAS)

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    async def close(self) -> None:
        await self.redis.aclose()

    async def create(self, kind: str, thread_id: str | None, request: dict[str, Any]) -> TaskView:
        now = time.time()
        job_id = uuid.uuid4().hex
        thread_id = thread_id or f"{kind}-{job_id}"
        fields = {
            "job_id": job_id, "kind": kind, "thread_id": thread_id,
            "created_at": now, "updated_at": now, "request": request,
        }
        mapping = {"status": "PENDING", **{k: json.dumps(v) for k, v in fields.items()}}
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self._key(job_id), mapping=mapping)
            pipe.expire(self._key(job_id), self.ttl)
            pipe.zadd(self.INDEX, {job_id: now})
            await pipe.execute()
        return (await self.get(job_id))  # type: ignore[return-value]

    async def get(self, job_id: str) -> TaskView | None:
        raw = await self.redis.hgetall(self._key(job_id))
        if not raw:
            return None
        data = {k: (v if k == "status" else json.loads(v)) for k, v in raw.items()}
        return TaskView(**data)

    async def update(self, job_id: str, **fields: Any) -> None:
        mapping: dict[str, str] = {}
        for key, value in {**fields, "updated_at": time.time()}.items():
            mapping[key] = value if key == "status" else json.dumps(value)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self._key(job_id), mapping=mapping)
            pipe.expire(self._key(job_id), self.ttl)
            await pipe.execute()

    async def claim(self, job_id: str, expected: str, new: str) -> int:
        """1 = transición hecha, 0 = el estado no era `expected`, -1 = el job no existe."""
        return int(await self._cas(keys=[self._key(job_id)], args=[expected, new, json.dumps(time.time())]))

    async def list(self, limit: int = 50) -> list[TaskView]:
        ids = await self.redis.zrevrange(self.INDEX, 0, limit - 1)
        jobs = [await self.get(i) for i in ids]
        expired = [i for i, j in zip(ids, jobs) if j is None]
        if expired:
            await self.redis.zrem(self.INDEX, *expired)
        return [j for j in jobs if j is not None]

    async def fail_stale(self) -> int:
        """Al arrancar: los jobs que quedaron RUNNING/PENDING ya no tienen worker => FAILED (evita polling infinito).
        Supone una sola instancia de la API. WAITING_APPROVAL se conserva: su estado vive en el checkpoint."""
        count = 0
        for job in await self.list(limit=1000):
            if job.status in ("PENDING", "RUNNING"):
                await self.update(job.job_id, status="FAILED", error="Interrumpido por reinicio del servicio.")
                count += 1
        return count
