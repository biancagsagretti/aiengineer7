import base64
from pathlib import Path
from typing import TypeVar

from openai import AsyncOpenAI
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import logger
from app.schemas.music import MusicInterpretation

T = TypeVar("T", bound=BaseModel)

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}

VISION_PROMPT = (
    "Analizá esta imagen que puede contener material para aprender guitarra. "
    "Describí solo lo que se observa: texto, diagramas de acordes, cuerdas, trastes, tablaturas, "
    "pentagramas, notas y símbolos. No inventes datos ilegibles. Indicá la incertidumbre. "
    "Respondé SOLO un JSON con las claves: content_type (text, chord_diagram, tablature, sheet_music, "
    "fretboard u other), description (en español), extracted_symbols (lista de strings), "
    "confidence (0 a 1) y needs_human_review (true/false). Contexto del documento: "
)


class LLMClient:
    """Adaptador async para OpenAI y Gemini (texto, visión, JSON validado y embeddings)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.normalized_provider
        self.openai = (
            AsyncOpenAI(
                api_key=settings.openai_api_key,
                max_retries=settings.llm_max_retries,
                timeout=settings.llm_timeout_seconds,
            )
            if settings.openai_api_key
            else None
        )
        self.gemini = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None

    @property
    def model_name(self) -> str:
        return self.settings.active_model

    # ------------------------------------------------------------------ texto
    async def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2, json_mode: bool = False) -> str:
        if self.provider == "openai":
            if not self.openai:
                raise RuntimeError("OPENAI_API_KEY no está configurada.")
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            result = await self.openai.chat.completions.create(
                model=self.settings.openai_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )
            return result.choices[0].message.content or ""
        if self.provider == "gemini":
            if not self.gemini:
                raise RuntimeError("GEMINI_API_KEY no está configurada.")
            config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json" if json_mode else None,
            )
            response = await self.gemini.aio.models.generate_content(
                model=self.settings.gemini_model,
                contents=f"{system_prompt}\n\nSolicitud del usuario:\n{user_prompt}",
                config=config,
            )
            return response.text or ""
        raise ValueError("LLM_PROVIDER debe ser 'openai' o 'gemini'.")

    async def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T | None:
        """Pide JSON y lo valida con Pydantic. Devuelve None si no es válido (el llamador decide el fallback)."""
        raw = await self.generate(system_prompt, user_prompt, temperature=0, json_mode=True)
        try:
            return schema.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            logger.warning("structured_output_invalid", schema=schema.__name__, error=str(exc)[:200])
            return None

    # ----------------------------------------------------------------- visión
    async def interpret_image(self, image_path: str, context: str = "") -> MusicInterpretation:
        path = Path(image_path)
        image_bytes = path.read_bytes()
        mime = _MIME.get(path.suffix.lower(), "image/png")
        prompt = VISION_PROMPT + context
        if self.provider == "openai":
            if not self.openai:
                raise RuntimeError("OPENAI_API_KEY no está configurada para visión.")
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            result = await self.openai.chat.completions.create(
                model=self.settings.openai_vision_model,
                response_format={"type": "json_object"},
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }],
            )
            raw = result.choices[0].message.content or ""
        elif self.provider == "gemini":
            if not self.gemini:
                raise RuntimeError("GEMINI_API_KEY no está configurada para visión.")
            response = await self.gemini.aio.models.generate_content(
                model=self.settings.gemini_vision_model,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
                config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
            )
            raw = response.text or ""
        else:
            raise ValueError("LLM_PROVIDER debe ser 'openai' o 'gemini'.")
        try:
            return MusicInterpretation.model_validate_json(raw)
        except (ValidationError, ValueError):
            # Fallback: se conserva el texto crudo marcado para revisión humana.
            return MusicInterpretation(content_type="other", description=raw.strip(), confidence=0.3, needs_human_review=True)

    async def describe_image(self, image_path: str, context: str = "") -> str:
        return (await self.interpret_image(image_path, context)).as_index_text()

    # ------------------------------------------------------------- embeddings
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.settings.embedding_provider.lower() != "openai":
            raise ValueError("Este sistema implementa embeddings OpenAI. EMBEDDING_PROVIDER debe ser openai.")
        if not self.openai:
            raise RuntimeError("OPENAI_API_KEY es necesaria para generar embeddings.")
        response = await self.openai.embeddings.create(model=self.settings.openai_embedding_model, input=texts)
        return [item.embedding for item in response.data]
