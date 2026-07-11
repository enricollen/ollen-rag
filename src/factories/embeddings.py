"""Provider-agnostic embedding factory: providers register a builder callable
(Settings -> BaseEmbedding) with EmbeddingFactory; selection comes from settings.
Concrete providers live in src/providers/ and self-register on import.
"""
from typing import Callable
from llama_index.core.embeddings import BaseEmbedding
from src.factories import model_catalog
from src.factories.model_catalog import EMBEDDING_MODELS_CONFIG_PATH, load_model_choices
from src.settings import Settings, get_settings

class EmbeddingFactory:
    """Registry mapping a provider name to a builder returning a llamaindex BaseEmbedding.

        @EmbeddingFactory.register("myprovider", model_field="myprovider_model_name")
        def build(settings: Settings) -> BaseEmbedding: ...

        model = EmbeddingFactory.create("myprovider", settings)

    model_field names the Settings attribute holding that provider's model id. It is a required
    argument: ingestion records an index's embedding model and retrieval restores it, and neither
    knows any provider by name, so a provider whose model cannot be located is unusable. The
    resolution logic itself lives in model_catalog, shared with RerankerFactory.
    """
    _registry: dict[str, Callable[[Settings], BaseEmbedding]] = {}
    _model_fields: dict[str, str] = {}

    @classmethod
    def register(cls, provider: str, model_field: str):
        """Function decorator registering *provider*'s builder and the field naming its model."""
        def decorator(builder: Callable[[Settings], BaseEmbedding]) -> Callable[[Settings], BaseEmbedding]:
            cls._registry[provider] = builder
            cls._model_fields[provider] = model_field
            return builder
        return decorator

    @classmethod
    def create(cls, provider: str, settings: Settings) -> BaseEmbedding:
        """Build the embedding model for *provider*, or raise listing known providers."""
        if provider not in cls._registry:
            raise ValueError(f"Unknown embedding provider '{provider}'. Available providers: {sorted(cls._registry)}")
        return cls._registry[provider](settings)

    @classmethod
    def providers(cls) -> list[str]:
        """Every registered provider name, sorted."""
        return sorted(cls._registry)

    @classmethod
    def model_field(cls, provider: str) -> str:
        """Name of the Settings field holding *provider*'s model id."""
        return model_catalog.model_field(cls._model_fields, provider)

    @classmethod
    def resolve_model(cls, settings: Settings, provider: str | None = None) -> str:
        """The model id configured for *provider* (default: the one settings selects)."""
        return model_catalog.resolve_model(settings, "embedding_provider", cls._model_fields, provider)

    @classmethod
    def with_model(cls, settings: Settings, provider: str, model: str | None = None) -> Settings:
        """A Settings copy pinned to *provider*, and to *model* when one is given."""
        return model_catalog.with_model(settings, "embedding_provider", cls._model_fields, provider, model)

    @classmethod
    def default_models(cls, settings: Settings) -> dict[str, str]:
        """provider -> its currently configured model, for the UI's /api/v1/config payload."""
        return model_catalog.default_models(settings, cls._model_fields)

# Cache of probed embedding dimensions keyed by model identifier
_DIM_CACHE: dict[str, int] = {}

def create_embedding_model(settings: Settings | None = None) -> BaseEmbedding:
    """Return the configured provider's llamaindex embedding model."""
    settings = settings or get_settings()
    # Local import: providers import this module for the registry, so a module-level
    # import here would be circular. This triggers provider self-registration once.
    import src.providers  # noqa: F401
    return EmbeddingFactory.create(settings.embedding_provider, settings)

def load_embedding_model_choices() -> dict[str, list[str]]:
    """Curated provider -> [model id, ...] list for the ingestion UI; static app data, not settings."""
    return load_model_choices(EMBEDDING_MODELS_CONFIG_PATH)

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
