from typing import Literal

from pydantic import BaseModel, Field

ContentType = Literal["text", "chord_diagram", "tablature", "sheet_music", "fretboard", "other"]


class MusicInterpretation(BaseModel):
    """Salida validada del modelo de visión para una imagen musical."""

    content_type: ContentType = "other"
    description: str
    extracted_symbols: list[str] = []
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = False

    def as_index_text(self) -> str:
        symbols = f"\nSímbolos: {', '.join(self.extracted_symbols)}" if self.extracted_symbols else ""
        return f"[{self.content_type}] {self.description}{symbols}"


class SupervisorDecision(BaseModel):
    intent: Literal["knowledge", "lesson", "evaluation", "general"]
    reasoning: str = ""
