import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from app.core.logging import logger
from app.rag.loaders import load_document


def is_daily_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "requests per day" in msg or "(rpd)" in msg or "insufficient_quota" in msg


class IngestionService:
    """Ingesta reanudable: guarda en disco qué texto/imágenes ya se indexaron."""

    def __init__(self, settings, llm, retriever):
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.quota_exhausted = False
        self._vision_calls = 0
        self._last_vision_call = 0.0
        self._lock = asyncio.Lock()  # una ingesta a la vez: evita corromper el estado
        self._state_path = Path(settings.processed_dir) / "ingestion_state.json"
        self._done: set[str] = self._load_state()

    # ---- estado persistente ---------------------------------------------
    def _load_state(self) -> set[str]:
        try:
            return set(json.loads(self._state_path.read_text(encoding="utf-8")))
        except Exception:
            return set()

    def _mark_done(self, key: str) -> None:
        self._done.add(key)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self._done)), encoding="utf-8")
        tmp.replace(self._state_path)  # escritura atómica

    @staticmethod
    def _text_key(text: str, metadata: dict) -> str:
        raw = f"{metadata.get('document')}|{metadata.get('page')}|{metadata.get('content_type')}|{text}"
        return "text:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()

    # ---- visión ----------------------------------------------------------
    async def _throttle(self) -> None:
        interval = self.settings.vision_min_interval
        if interval <= 0:
            return
        wait = interval - (time.monotonic() - self._last_vision_call)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_vision_call = time.monotonic()

    def _should_analyze(self, record: dict) -> bool:
        if not self.settings.vision_enabled:
            return False
        if record["metadata"].get("content_type") == "page_image" and not self.settings.vision_page_renders:
            return False
        return True

    # ---- ingesta ---------------------------------------------------------
    @property
    def vision_calls_used(self) -> int:
        return self._vision_calls

    async def plan_path(self, path: Path) -> dict:
        """Estima el trabajo pendiente SIN llamar a ningún modelo (no gasta tokens)."""
        image_dir = Path(self.settings.processed_dir) / "images"
        records = await asyncio.to_thread(load_document, path, image_dir)
        text_pending = images_total = images_pending = images_small = 0
        for record in records:
            text = record.get("text", "")
            if text.strip() and self._text_key(text, record["metadata"]) not in self._done:
                text_pending += 1
            image_path = record.get("image_path")
            if not image_path:
                continue
            images_total += 1
            if f"img:{image_path}" in self._done or not self._should_analyze(record):
                continue
            try:
                if os.path.getsize(image_path) < self.settings.vision_min_image_bytes:
                    images_small += 1
                    continue
            except OSError:
                continue
            images_pending += 1
        cap = self.settings.vision_max_per_run
        estimated = min(images_pending, cap) if cap else images_pending
        return {
            "text_records_pending": text_pending,
            "images_total": images_total,
            "images_small_skipped": images_small,
            "images_pending": images_pending,
            "estimated_vision_calls": estimated,
        }

    async def ingest_path(
        self, path: Path, *, reset_counters: bool = True, vision_budget: int | None = None
    ) -> tuple[int, list[str]]:
        """vision_budget: tope DURO de llamadas de visión (0 = ninguna). None = solo aplica VISION_MAX_PER_RUN."""
        async with self._lock:
            return await self._ingest_path(path, reset_counters, vision_budget)

    async def _ingest_path(self, path: Path, reset_counters: bool, vision_budget: int | None) -> tuple[int, list[str]]:
        limits = [x for x in (self.settings.vision_max_per_run or None, vision_budget) if x is not None]
        limit = min(limits) if limits else None
        if reset_counters:
            self.quota_exhausted = False
            self._vision_calls = 0
        image_dir = Path(self.settings.processed_dir) / "images"
        # La extracción (PyMuPDF/python-docx) es CPU/disco síncrono: se saca del event loop.
        records = await asyncio.to_thread(load_document, path, image_dir)
        indexed, warnings, skipped_images = 0, [], 0

        for record in records:
            metadata = record["metadata"]
            text = record.get("text", "")
            if text.strip():
                key = self._text_key(text, metadata)
                if key not in self._done:
                    indexed += await self.retriever.index_text(text, metadata)
                    self._mark_done(key)

            image_path = record.get("image_path")
            if not image_path:
                continue
            img_key = f"img:{image_path}"
            if img_key in self._done:
                continue
            if not self._should_analyze(record) or self.quota_exhausted:
                skipped_images += 1
                continue
            if limit is not None and self._vision_calls >= limit:
                skipped_images += 1
                continue
            try:
                if os.path.getsize(image_path) < self.settings.vision_min_image_bytes:
                    self._mark_done(img_key)  # decorativa: no vale la pena gastar tokens
                    continue
            except OSError:
                continue

            try:
                await self._throttle()
                self._vision_calls += 1
                interp = await self.llm.interpret_image(
                    image_path,
                    context=f"Documento {metadata.get('document')}, página {metadata.get('page')}",
                )
                img_meta = {
                    **metadata,
                    "image_path": image_path,
                    "music_type": interp.content_type,
                    "vision_confidence": interp.confidence,
                    "needs_human_review": interp.needs_human_review,
                }
                indexed += await self.retriever.index_text(interp.as_index_text(), img_meta)
                self._mark_done(img_key)
            except Exception as exc:
                if is_daily_quota_error(exc):
                    self.quota_exhausted = True
                    logger.error("daily_quota_exceeded_stopping", path=image_path)
                    warnings.append(
                        "Límite diario de OpenAI agotado: las imágenes restantes quedaron pendientes. "
                        "Vuelve a ejecutar la ingesta más tarde y continuará donde quedó."
                    )
                    continue
                warnings.append(f"No se pudo analizar {image_path}: {exc}")
                logger.warning("image_analysis_failed", path=image_path, error=str(exc))

        if skipped_images:
            warnings.append(f"{skipped_images} imagen(es) pendientes de análisis en {path.name} (visión desactivada, tope o cuota).")
        return indexed, warnings

    async def ingest_folder(self) -> tuple[int, int, list[str]]:
        folder = Path(self.settings.documents_dir)
        folder.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".docx"})
        total_chunks, processed, warnings = 0, 0, []
        self.quota_exhausted = False
        self._vision_calls = 0
        for path in files:
            try:
                chunks, file_warnings = await self.ingest_path(path, reset_counters=False)
                total_chunks += chunks
                warnings.extend(file_warnings)
                processed += 1
                if self.quota_exhausted:
                    break
            except Exception as exc:
                warnings.append(f"{path.name}: {exc}")
        return processed, total_chunks, warnings
