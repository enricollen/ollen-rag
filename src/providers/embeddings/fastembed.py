"""fastembed provider: thin llamaindex adapter around fastembed's ONNX CPU TextEmbedding.

The official llama-index-embeddings-fastembed package requires Python <3.13,
so this project wraps the fastembed library directly.
"""
from typing import Any
from fastembed import TextEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from src.factories.embeddings import EmbeddingFactory
from src.settings import Settings

class FastEmbedEmbedding(BaseEmbedding):
    """CPU embedding model backed by fastembed (ONNX), no GPU or network calls at inference."""
    _model: Any = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None, **kwargs: Any) -> None:
        # Initialize the pydantic base first, then attach the private fastembed model instance.
        super().__init__(model_name=model_name, **kwargs)
        # cache_dir persists the ONNX model files locally (default fastembed cache is a temp dir
        # that re-downloads); pointing it under models/ keeps embeddings coherent with the reranker.
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @classmethod
    def class_name(cls) -> str:
        """Identifier used by llamaindex serialization."""
        return "FastEmbedEmbedding"

    def _get_text_embedding(self, text: str) -> list[float]:
        # fastembed returns a generator of numpy arrays; cast to plain floats for JSON safety
        return [float(x) for x in next(iter(self._model.embed([text])))]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Batch variant of the above, used for indexing multiple chunks at once.
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def _get_query_embedding(self, query: str) -> list[float]:
        # Queries use fastembed's dedicated query_embed (may apply query-side prompt prefixes).
        return [float(x) for x in next(iter(self._model.query_embed(query)))]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        # fastembed has no native async API; delegate to the sync implementation.
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        # fastembed has no native async API; delegate to the sync implementation.
        return self._get_text_embedding(text)

@EmbeddingFactory.register("fastembed", model_field="fastembed_model_name")
def create_fastembed_embedding(settings: Settings) -> BaseEmbedding:
    """Registered builder: CPU embeddings from the configured fastembed model."""
    return FastEmbedEmbedding(model_name=settings.fastembed_model_name, cache_dir=settings.fastembed_cache_dir)