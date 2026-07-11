"""Provider-agnostic reranker: each provider (in src/providers/reranker/) implements
RerankConnector.rerank() and self-registers with RerankerFactory; ConnectorRerank adapts whichever
connector is configured to the llama_index postprocessor interface that CitationQueryEngine needs.

Mirrors src/factories/llm.py.
"""
import math
from abc import ABC, abstractmethod
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from src.factories import model_catalog
from src.factories.model_catalog import RERANKER_MODELS_CONFIG_PATH, load_model_choices, validate_model
from src.settings import Settings, get_settings

def to_probability(logit: float) -> float:
    """Map a cross-encoder logit to the 0-1 relevance probability it was trained to predict.

    These rerankers are single-label BertForSequenceClassification models trained with binary
    cross-entropy, so sigmoid() is their calibrated output. Monotonic, so ranking is untouched.
    Computed branch-wise to stay finite for large-magnitude logits, where a naive 1/(1+exp(-x))
    overflows.

    Lives here rather than in one connector because more than one provider emits logits: the local
    cross-encoder does, and so does watsonx's rerank endpoint (see LiteLLMWatsonxRerankConnector).
    """
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    odds = math.exp(logit)
    return odds / (1.0 + odds)

class RerankConnector(ABC):
    """Provider-agnostic reranker interface: implement rerank() and register with RerankerFactory.

    Contract: return at most *top_n* NodeWithScore, sorted by descending score, where score is a
    calibrated 0-1 relevance probability.

    Normalizing inside the connector is what lets a local cross-encoder (which emits raw logits)
    and a Cohere-style HTTP API (which returns relevance_score already in 0-1) coexist. Before this
    contract existed, retrieval pinned every cross-encoder to Identity and generation applied the
    sigmoid twenty lines away in another module -- so /retrieve and the MCP tool leaked raw logits
    while /generate returned probabilities.
    """
    model_name: str = "connector-rerank"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @abstractmethod
    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        """Score *nodes* against *query* and return the top_n best, best first."""

    def warmup(self) -> None:
        """Pre-load whatever the first rerank() would otherwise load lazily.

        Called once at app startup. A no-op for HTTP-backed providers, which have nothing local to
        load; the cross-encoder connector overrides it to pull model weights off disk up front so
        the first query does not pay for them.
        """

class RerankerFactory:
    """Registry mapping a provider name to its RerankConnector class.

        @RerankerFactory.register("myprovider", model_field="myprovider_rerank_model")
        class MyConnector(RerankConnector): ...

        connector = RerankerFactory.create("myprovider", settings=settings)

    model_field names the Settings attribute holding that provider's model id, and is required for
    the same reason it is on EmbeddingFactory: a per-request override has to land in the field the
    provider actually reads. The resolution logic lives in model_catalog, shared between the two.
    """
    _registry: dict[str, type[RerankConnector]] = {}
    _model_fields: dict[str, str] = {}

    @classmethod
    def register(cls, provider: str, model_field: str):
        """Class decorator registering *provider*'s connector and the field naming its model."""
        def decorator(connector_cls: type[RerankConnector]) -> type[RerankConnector]:
            cls._registry[provider] = connector_cls
            cls._model_fields[provider] = model_field
            return connector_cls
        return decorator

    @classmethod
    def create(cls, provider: str, **kwargs) -> RerankConnector:
        """Instantiate the connector for *provider*, or raise listing known providers."""
        cls.ensure_registered(provider)
        return cls._registry[provider](**kwargs)

    @classmethod
    def ensure_registered(cls, provider: str) -> None:
        """Raise if *provider* was never registered. The registry -- not the yaml catalog -- is the
        source of truth for which providers exist, so request handlers validate names against it."""
        if provider not in cls._registry:
            raise ValueError(f"Unknown reranker provider '{provider}'. Available providers: {sorted(cls._registry)}")

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
        return model_catalog.resolve_model(settings, "reranker_provider", cls._model_fields, provider)

    @classmethod
    def with_model(cls, settings: Settings, provider: str, model: str | None = None) -> Settings:
        """A Settings copy pinned to *provider*, and to *model* when one is given."""
        return model_catalog.with_model(settings, "reranker_provider", cls._model_fields, provider, model)

    @classmethod
    def default_models(cls, settings: Settings) -> dict[str, str]:
        """provider -> its currently configured model, for the UI's /api/v1/config payload."""
        return model_catalog.default_models(settings, cls._model_fields)

class ConnectorRerank(BaseNodePostprocessor):
    """Adapts any RerankConnector to llama_index's postprocessor interface, so the query engine
    stays provider-blind: only this class knows about llama_index's calling convention."""
    connector: RerankConnector
    top_n: int

    @classmethod
    def class_name(cls) -> str:
        """Identifier used by llamaindex serialization."""
        return "ConnectorRerank"

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None) -> list[NodeWithScore]:
        """Forward to the connector, wrapped in the RERANKING callback event llamaindex expects."""
        if query_bundle is None:
            raise ValueError("Reranking requires a query.")
        if not nodes:
            return []
        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={EventPayload.NODES: nodes, EventPayload.MODEL_NAME: self.connector.model_name,
                     EventPayload.QUERY_STR: query_bundle.query_str, EventPayload.TOP_K: self.top_n},
        ) as event:
            new_nodes = self.connector.rerank(query_bundle.query_str, nodes, self.top_n)
            event.on_end(payload={EventPayload.NODES: new_nodes})
        return new_nodes

def load_reranker_model_choices() -> dict[str, list[str]]:
    """Curated provider -> [model id, ...] list for the Retrieval/Query UI; static app data, not settings."""
    return load_model_choices(RERANKER_MODELS_CONFIG_PATH)

# Connectors are heavy to build (a cross-encoder loads model weights); keep one per
# (provider, model) for the process lifetime, as the old _rerankers dict did.
_connectors: dict[tuple[str, str], RerankConnector] = {}

def _get_connector(provider: str, model: str | None, settings: Settings) -> RerankConnector:
    """Return the shared connector for (provider, model), building it on first use."""
    resolved = model or RerankerFactory.resolve_model(settings, provider)
    key = (provider, resolved)
    if key not in _connectors:
        scoped = RerankerFactory.with_model(settings, provider, resolved)
        _connectors[key] = RerankerFactory.create(provider, settings=scoped)
    return _connectors[key]

def create_reranker(
    top_n: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> ConnectorRerank:
    """Return the configured provider's connector, adapted to the llama_index postprocessor interface.

    provider/model default to settings; passing either overrides it for this call only. First use
    of a not-yet-cached (provider, model) pays its construction cost once.
    """
    settings = settings or get_settings()
    # Local import: providers import this module for the registry, so a module-level
    # import here would be circular. This triggers provider self-registration once.
    import src.providers  # noqa: F401
    provider = provider or settings.reranker_provider
    # The registry decides which providers exist; the yaml catalog only constrains which models a
    # caller may request for one. Validating the name against the yaml instead would reject any
    # provider registered from outside this repo.
    RerankerFactory.ensure_registered(provider)
    if model:
        validate_model(load_reranker_model_choices(), provider, model)
    connector = _get_connector(provider, model, settings)
    return ConnectorRerank(connector=connector, top_n=top_n or settings.rerank_top_n)
