"""Prueba de carga: N peticiones concurrentes contra POST /v1/tasks y espera a que terminen.

Uso (dentro del contenedor, sin instalar nada):
    docker compose exec api python scripts/load_test.py
Desde tu máquina:   pip install httpx   y   python scripts/load_test.py --base-url http://localhost:8000

Mide la latencia desde el cliente (encolado + ejecución + polling). El costo por ejecución y la latencia p95
OFICIALES para la entrega se leen en el dashboard de Phoenix (http://localhost:6006).
"""
import argparse
import asyncio
import math
import os
import statistics
import time
import uuid

import httpx

QUESTIONS = [
    "¿Cómo practico el cambio entre Do mayor y Sol mayor?",
    "Explicame cómo se toca un acorde de La menor y qué errores evitar.",
    "Armame una rutina de 20 minutos para mejorar la técnica de púa alterna.",
    "Evaluame con preguntas sobre las notas en las primeras cuerdas de la guitarra.",
    "¿Qué ejercicios me recomendás para fortalecer los dedos de la mano izquierda?",
]
DONE = {"DONE", "FAILED", "REJECTED"}


async def run_one(client: httpx.AsyncClient, base: str, idx: int, question: str, run_id: str, poll: float, timeout: float) -> dict:
    start = time.perf_counter()
    body = {"question": question, "session_id": f"load-{run_id}-{idx}", "level": "principiante"}
    resp = await client.post(f"{base}/v1/tasks", json=body)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    status, error = "PENDING", None
    while time.perf_counter() - start < timeout:
        job = (await client.get(f"{base}/v1/tasks/{job_id}")).json()
        status, error = job["status"], job.get("error")
        if status in DONE:
            break
        await asyncio.sleep(poll)
    else:
        status, error = "TIMEOUT", f"sin terminar tras {timeout}s"
    return {"n": idx, "job_id": job_id, "status": status, "seconds": time.perf_counter() - start, "error": error}


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank: con 5 muestras, p95 = la más lenta."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    run_id = uuid.uuid4().hex[:6]
    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(args.requests)]
    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            run_one(client, args.base_url, i + 1, q, run_id, args.poll, args.timeout) for i, q in enumerate(questions)
        ])
        wall = time.perf_counter() - t0

    print(f"\nCorrida {run_id}: {args.requests} peticiones concurrentes, {wall:.1f}s en total\n")
    for r in results:
        print(f"  #{r['n']}  {r['status']:<8} {r['seconds']:6.1f}s  job={r['job_id']}" + (f"  error={r['error']}" if r["error"] else ""))
    ok = [r["seconds"] for r in results if r["status"] == "DONE"]
    if ok:
        print(f"\n  OK: {len(ok)}/{len(results)} | media {statistics.mean(ok):.1f}s | p95 (cliente) {percentile(ok, 95):.1f}s | máx {max(ok):.1f}s")
    print("\nAhora abrí http://localhost:6006 -> proyecto 'guitar-intelligence-system' y capturá costo por ejecución y latencia p95.")


if __name__ == "__main__":
    asyncio.run(main())
