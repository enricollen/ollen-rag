"""Local cross-encoder reranker, self-registered with the reranker factory on import.

Calls sentence_transformers.CrossEncoder directly rather than going through llamaindex's
SentenceTransformerRerank, so the activation function is pinned at the call site.
"""
from functools import lru_cache
import torch
from llama_index.core.schema import MetadataMode, NodeWithScore
from sentence_transformers import CrossEncoder
from src.factories.reranker import RerankConnector, RerankerFactory
from src.settings import Settings

@lru_cache
def _cross_encoder(model_id: str) -> CrossEncoder:
    """One CrossEncoder per model id for the process lifetime; loading weights is expensive."""
    return CrossEncoder(model_id)

@RerankerFactory.register("sentence-transformers", model_field="reranker_model")
class SentenceTransformerRerankConnector(RerankConnector):
    """Single-label cross-encoder scoring (query, passage) pairs on CPU/GPU locally.

    activation_fn=Sigmoid is passed explicitly on every call. sentence-transformers otherwise
    chooses per model -- models/reranker ships a config_sentence_transformers.json declaring
    Identity, while BAAI/bge-reranker-v2-m3 ships none and falls back to Sigmoid -- so node.score
    would be a logit for one model and a probability for another. These models are trained with
    binary cross-entropy, so sigmoid is their calibrated output.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.model_name = self._settings.reranker_model

    def warmup(self) -> None:
        """Load the model weights now (cached), so the first query does not pay for them."""
        _cross_encoder(self.model_name)

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        """Score every node against the query, then return the top_n best, best first."""
        pairs = [(query, node.node.get_content(metadata_mode=MetadataMode.EMBED)) for node in nodes]
        scores = _cross_encoder(self.model_name).predict(pairs, activation_fn=torch.nn.Sigmoid())
        # float() casts off numpy/torch scalar types so the API layer can JSON-serialize the score.
        # Fresh NodeWithScore wrappers: the caller's fused scores must survive this call untouched.
        ranked = [NodeWithScore(node=node.node, score=float(score)) for node, score in zip(nodes, scores, strict=True)]
        ranked.sort(key=lambda node: node.score, reverse=True)
        return ranked[:top_n]
