from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    metadata: dict


def split_text(text: str, metadata: dict, chunk_size: int = 1200, overlap: int = 180) -> list[TextChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Se requiere chunk_size > overlap >= 0.")
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunks.append(TextChunk(text=cleaned[start:end], metadata={**metadata, "chunk_start": start}))
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks
