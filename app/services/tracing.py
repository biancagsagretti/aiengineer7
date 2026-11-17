from contextlib import contextmanager

from app.core.config import Settings
from app.core.logging import logger

_tracer = None


def initialize_tracing(settings: Settings) -> None:
    """Configura OpenTelemetry -> Arize Phoenix por HTTP (puerto 6006).

    Instrumenta LangChain/LangGraph (nodos del grafo) y el SDK de OpenAI (cada
    llamada al modelo y a embeddings), de modo que la traza muestre todo el camino.
    """
    global _tracer
    if not settings.phoenix_enabled:
        logger.info("phoenix_tracing_disabled")
        return
    try:
        from phoenix.otel import register

        endpoint = settings.phoenix_collector_endpoint.rstrip("/") + "/v1/traces"
        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=endpoint,
            protocol="http/protobuf",
            batch=True,  # BatchSpanProcessor: no bloquea las peticiones
            auto_instrument=False,  # instrumentamos a mano LangChain/LangGraph y OpenAI; sin ruido de FastAPI
            set_global_tracer_provider=True,
        )
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        except ImportError:
            logger.warning("openai_instrumentation_missing", hint="pip install openinference-instrumentation-openai")
        _tracer = tracer_provider.get_tracer("guitar-intelligence-system")
        logger.info("phoenix_tracing_initialized", endpoint=endpoint)
    except Exception as exc:
        # La API debe poder arrancar aunque Phoenix esté caído.
        logger.warning("phoenix_tracing_initialization_failed", error=str(exc))


@contextmanager
def span(name: str, new_root: bool = False, **attributes):
    """Span manual. new_root=True crea una traza propia aunque haya un span activo (p. ej. la petición HTTP
    que encoló la tarea): así la latencia y el costo de la traza son los de TODA la ejecución del agente."""
    if _tracer is None:
        yield None
        return
    kwargs = {}
    if new_root:
        from opentelemetry.context import Context

        kwargs["context"] = Context()
    with _tracer.start_as_current_span(name, **kwargs) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
        yield current
