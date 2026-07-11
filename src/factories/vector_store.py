"""Agnostic vector store layer: query modes, the VectorStoreBackend interface + registry,
backend selection, and the index-naming helper. Concrete backends live in
src/providers/vector_stores/ and self-register on import."""
from abc import ABC, abstractmethod
from enum import Enum
from llama_index.core.schema import BaseNode, NodeWithScore
from src.exceptions import VectorStoreError
from src.logger import OllenLogger
from src.settings import Settings, get_settings

log = OllenLogger("vector_store")

class QueryMode(str, Enum):
    """Retrieval mode requested of a backend. Our own enum so llamaindex's
    VectorStoreQueryMode never leaks past a backend's internals."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

# Preference order when a requested mode is unsupported: richest first.
_MODE_PRIORITY = (QueryMode.HYBRID, QueryMode.DENSE, QueryMode.SPARSE)

class VectorStoreBackend(ABC):
    """One vector database behind a store-agnostic interface. Concrete backends
    live in src/providers/vector_stores/<name>.py and self-register."""

    @property
    @abstractmethod
    def supported_query_modes(self) -> set["QueryMode"]:
        """Query modes this backend can actually serve."""

    def warmup(self) -> None:
        """Optional one-time startup priming (e.g. create a shared search pipeline). No-op by default."""

    @abstractmethod
    def ensure_ready(self, index: str, dim: int) -> None:
        """Idempotently create the index/collection and any store-side setup."""

    @abstractmethod
    def add_nodes(self, index: str, nodes: list[BaseNode]) -> None:
        """Write already-embedded nodes into the index."""

    @abstractmethod
    def retrieve(self, index: str, query_str: str, query_embedding: list[float],
                 mode: "QueryMode", top_k: int,
                 raw_filters: list[dict] | None, filter_condition: str) -> list[NodeWithScore]:
        """Return raw, scored nodes for the given mode. Threshold + rerank are applied by the caller."""

    @abstractmethod
    def get_index_meta(self, index: str) -> dict | None:
        """Return the recorded build config (embedding + chunking) for an index, or None."""

    @abstractmethod
    def set_index_meta(self, index: str, embedding_provider: str, embedding_model: str, chunking: dict) -> None:
        """Record the index's build config."""

    @abstractmethod
    def get_index_dim(self, index: str) -> int | None:
        """Return the index's stored vector dimension, or None if it does not exist."""

    @abstractmethod
    def list_indices(self) -> list[dict]:
        """Service-owned indices with document counts."""

    @abstractmethod
    def get_index_documents(self, index: str, offset: int, limit: int, bucket: str | None = None) -> dict:
        """Paginate stored documents (content + metadata) for browsing, optionally scoped to one bucket."""

    @abstractmethod
    def list_buckets(self, index: str) -> list[str]:
        """Distinct metadata.bucket values present in an index."""

    @abstractmethod
    def list_bucket_files(self, index: str) -> dict[str, list[str]]:
        """Map each bucket to the distinct file_names it contains."""

    @abstractmethod
    def find_duplicate_file(self, index: str, file_hash: str, bucket: str | None) -> str | None:
        """file_name of an already-indexed doc with the same hash+bucket, or None."""

    @abstractmethod
    def delete_index(self, index: str) -> None:
        """Permanently delete an index and all its documents."""

    @abstractmethod
    def delete_bucket(self, index: str, bucket: str) -> int:
        """Delete every document in `index` whose metadata.bucket == `bucket`; return the count deleted. Idempotent: missing index/bucket returns 0."""

class VectorStoreFactory:
    """Decorator registry mapping a backend name to its VectorStoreBackend class."""
    _registry: dict[str, type[VectorStoreBackend]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator registering a backend under *name*."""
        def decorator(klass: type[VectorStoreBackend]) -> type[VectorStoreBackend]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, settings: Settings) -> VectorStoreBackend:
        """Instantiate the backend registered under *name*, or raise ValueError."""
        if name not in cls._registry:
            raise ValueError(f"Unknown vector store '{name}'. Available: {sorted(cls._registry)}")
        return cls._registry[name](settings)

def create_backend(settings: Settings | None = None) -> VectorStoreBackend:
    """Return the configured vector store backend (settings.vector_store).

    Lazy-imports src.providers for registration side effects, exactly like create_llm()."""
    settings = settings or get_settings()
    import src.providers  # noqa: F401  triggers backend self-registration
    return VectorStoreFactory.create(settings.vector_store, settings)

def embedding_meta(backend: VectorStoreBackend, index: str) -> dict | None:
    """Embedding-only view over an index's recorded meta: {embedding_provider, embedding_model} or None."""
    meta = backend.get_index_meta(index)
    if not meta or not meta.get("embedding_provider"):
        return None
    return {"embedding_provider": meta["embedding_provider"], "embedding_model": meta["embedding_model"]}

def pick_supported_mode(backend: VectorStoreBackend, requested: QueryMode) -> QueryMode:
    """Return *requested* if the backend supports it, else the richest supported mode (with a warning)."""
    if requested in backend.supported_query_modes:
        return requested
    for mode in _MODE_PRIORITY:
        if mode in backend.supported_query_modes:
            log.warning("backend does not support %s; falling back to %s", requested.value, mode.value)
            return mode
    raise VectorStoreError("backend declares no supported query modes")

def build_index_name(strategy: str | None, index_name: str | None, settings: Settings | None = None) -> str:
    """Explicit index name wins; otherwise '{prefix}_{strategy}' (default strategy if omitted)."""
    settings = settings or get_settings()
    if index_name:
        return index_name
    return f"{settings.opensearch_index_prefix}_{strategy or settings.default_chunking_strategy}"