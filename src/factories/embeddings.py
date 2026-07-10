"""Provider-agnostic embedding factory: providers register a builder callable
(Settings -> BaseEmbedding) with EmbeddingFactory; selection comes from settings.
Concrete providers live in src/providers/ and self-register on import.
"""
from functools import lru_cache
from pathlib import Path
from typing import Callable
import yaml
from llama_index.core.embeddings import BaseEmbedding
from src.settings import Settings, get_settings

class EmbeddingFactory:
    """Registry mapping a provider name to a builder returning a llamaindex BaseEmbedding.

        @EmbeddingFactory.register("myprovider")
        def build(settings: Settings) -> BaseEmbedding: ...

        model = EmbeddingFactory.create("myprovider", settings)
    """
    _registry: dict[str, Callable[[Settings], BaseEmbedding]] = {}

    @classmethod
    def register(cls, provider: str):
        """Class decorator/function decorator registering *provider*'s builder callable."""
        def decorator(builder: Callable[[Settings], BaseEmbedding]) -> Callable[[Settings], BaseEmbedding]:
            cls._registry[provider] = builder
            return builder
        return decorator

    @classmethod
    def create(cls, provider: str, settings: Settings) -> BaseEmbedding:
        """Build the embedding model for *provider*, or raise listing known providers."""
        if provider not in cls._registry:
            raise ValueError(f"Unknown embedding provider '{provider}'. Available providers: {sorted(cls._registry)}")
        return cls._registry[provider](settings)

# Cache of probed embedding dimensions keyed by model identifier
_DIM_CACHE: dict[str, int] = {}

def create_embedding_model(settings: Settings | None = None) -> BaseEmbedding:
    """Return the configured provider's llamaindex embedding model."""
    settings = settings or get_settings()
    # Local import: providers import this module for the registry, so a module-level
    # import here would be circular. This triggers provider self-registration once.
    import src.providers  # noqa: F401
    return EmbeddingFactory.create(settings.embedding_provider, settings)

EMBEDDING_MODELS_CONFIG_PATH = Path("config/embedding_models.yaml")

@lru_cache
def load_embedding_model_choices() -> dict[str, list[str]]:
    """Curated provider -> [model id, ...] list for the ingestion UI; static app data, not settings."""
    return yaml.safe_load(EMBEDDING_MODELS_CONFIG_PATH.read_text(encoding="utf-8"))

def get_embedding_dim(embed_model: BaseEmbedding) -> int:
    """Probe the model once with a tiny input to discover its vector dimension (cached).
    Cache key prioritizes model_id (some providers only set that), then model_name, then class name."""
    key = (
        getattr(embed_model, "model_id", None)
        or getattr(embed_model, "model_name", None)
        or embed_model.__class__.__name__
    )
    if key not in _DIM_CACHE:
        _DIM_CACHE[key] = len(embed_model.get_text_embedding("dimension probe"))
    return _DIM_CACHE[key]