from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Guitar Intelligence System"
    app_env: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    api_key: str = ""

    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_vision_model: str = "gemini-2.5-flash"
    embedding_provider: str = "openai"
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 60.0

    documents_dir: str = "./data/documents"
    processed_dir: str = "./data/processed"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "guitar_knowledge"
    vector_size: int = 1536
    top_k: int = 6
    chunk_size: int = 1200
    chunk_overlap: int = 180

    # --- Visión / ingesta de imágenes (controla el gasto de tokens) ---
    vision_enabled: bool = True           # false = ingesta solo de texto
    vision_page_renders: bool = True      # analizar el render de cada página (puede duplicar costo)
    vision_min_interval: float = 0.0      # segundos entre llamadas de visión (Free tier 3 RPM => 20)
    vision_max_per_run: int = 0           # 0 = sin tope por ejecución
    vision_min_image_bytes: int = 8000    # imágenes más chicas se descartan

    # --- Redis: checkpoints de LangGraph + estado de los jobs ---
    redis_url: str = "redis://localhost:6379/0"
    checkpointer_backend: str = "redis"   # "redis" | "memory" (solo desarrollo/tests)
    job_ttl_seconds: int = 7 * 24 * 3600
    worker_concurrency: int = 5

    # --- Human-in-the-loop: la ingesta con visión es la acción "crítica" ---
    hitl_min_vision_calls: int = 1        # >= N llamadas de visión estimadas => pide aprobación humana
    # Supuestos para ESTIMAR el costo (editables; verificá los precios vigentes de tu modelo):
    vision_est_input_tokens: int = 1800   # tokens de entrada por imagen (imagen + prompt)
    vision_est_output_tokens: int = 350
    vision_price_in_per_mtok: float = 0.40
    vision_price_out_per_mtok: float = 1.60

    phoenix_enabled: bool = True
    # Base HTTP de Phoenix (sin /v1/traces). En Docker: http://phoenix:6006
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "guitar-intelligence-system"

    @property
    def normalized_provider(self) -> str:
        return self.llm_provider.strip().lower()

    @property
    def active_model(self) -> str:
        return self.openai_model if self.normalized_provider == "openai" else self.gemini_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
