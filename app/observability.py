"""Punto de entrada de observabilidad (Arize Phoenix vía OpenInference). Ver app/services/tracing.py."""
from app.services.tracing import initialize_tracing, span

__all__ = ["initialize_tracing", "span"]
