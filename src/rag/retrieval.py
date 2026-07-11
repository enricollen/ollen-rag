"""Hybrid retrieval through the configured vector store backend, with metadata filters,
fused-score floor, and cross-encoder reranking."""
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from src.exceptions import VectorStoreError
from src.factories.embeddings import EmbeddingFactory, create_embedding_model
from src.factories.reranker import create_reranker
from src.factories.vector_store import (
    QueryMode, build_index_name, create_backend, embedding_meta, pick_supported_mode,
)
from src.logger import OllenLogger
from src.settings import get_settings

log = OllenLogger("retrieval")

def _resolve_embed_settings(settings, index: str, backend):
    """Return a Settings copy whose embedding provider/model match whatever built *index*.

    Querying with a different model than the index's vectors silently mixes vector spaces,
    so we read the model recorded in the index meta (via the backend), never the global default.
    """
    meta = embedding_meta(backend, index)
    if meta:
        return EmbeddingFactory.with_model(settings, meta["embedding_provider"], meta["embedding_model"])
    log.warning(
        "index '%s' has no recorded embedding metadata; using currently configured default (%s/%s)",
        index, settings.embedding_provider, EmbeddingFactory.resolve_model(settings),
    )
    return settings

def _query_nodes(backend, index, query, mode, top_k, raw_filters, filter_condition, settings):
    """Embed the query with the index's model and run one backend retrieve in *mode* (with fallback)."""
    embed_model = create_embedding_model(settings)
    query_embedding = embed_model.get_query_embedding(query)
    effective_mode = pick_supported_mode(backend, mode)
    return backend.retrieve(index, query, query_embedding, effective_mode, top_k, raw_filters, filter_condition)

class BackendRetriever(BaseRetriever):
    """llamaindex retriever adapter that runs a hybrid query through a VectorStoreBackend.

    Lets llamaindex constructs (e.g. CitationQueryEngine) drive any backend without a
    VectorStoreIndex. Returns the raw fused nodes; threshold/rerank stay engine postprocessors,
    matching the pre-refactor build_retriever behavior.
    """
    def __init__(self, backend, index: str, settings, top_k: int, raw_filters: list[dict] | None, filter_condition: str):
        self._backend = backend
        self._index = index
        self._settings = settings
        self._top_k = top_k
        self._raw_filters = raw_filters
        self._filter_condition = filter_condition
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Run the hybrid backend query for the bundle's query string."""
        return _query_nodes(
            self._backend, self._index, query_bundle.query_str, QueryMode.HYBRID,
            self._top_k, self._raw_filters, self._filter_condition, self._settings,
        )

def build_backend_retriever(
    index_name: str, top_k: int, raw_filters: list[dict] | None = None, filter_condition: str = "and",
) -> BaseRetriever:
    """Return a llamaindex retriever bound to the configured backend and the index's embed model."""
    settings = get_settings()
    backend = create_backend(settings)
    settings = _resolve_embed_settings(settings, index_name, backend)
    return BackendRetriever(backend, index_name, settings, top_k, raw_filters, filter_condition)

def get_threshold_postprocessor(threshold: float | None = None) -> SimilarityPostprocessor | None:
    """Score floor on fused hybrid scores (min_max-normalized, 0-1); None falls back to
    settings, 0 disables. Never applied to rerank scores, which live on a different
    (cross-encoder) scale."""
    effective = get_settings().similarity_threshold if threshold is None else threshold
    if not effective:
        return None
    return SimilarityPostprocessor(similarity_cutoff=effective)

def retrieve(
    query: str,
    strategy: str | None = None,
    index_name: str | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    raw_filters: list[dict] | None = None,
    filter_condition: str = "and",
    similarity_threshold: float | None = None,
    use_rerank: bool = True,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> list[NodeWithScore]:
    """Hybrid retrieve, fused-score floor, then rerank: returns the rerank_top_n best nodes for the query.

    Reranked node scores are 0-1 relevance probabilities (see RerankConnector's contract), not the
    fused hybrid scores the threshold operates on.
    """
    settings = get_settings()
    backend = create_backend(settings)
    target_index = build_index_name(strategy, index_name, settings)
    k = top_k or settings.retrieval_top_k
    settings = _resolve_embed_settings(settings, target_index, backend)
    try:
        nodes = _query_nodes(backend, target_index, query, QueryMode.HYBRID, k, raw_filters, filter_condition, settings)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"Retrieval failed on index '{target_index}': {exc}") from exc
    if not nodes:
        log.info("retrieve: index=%s top_k=%d filters=%d -> 0 node(s)", target_index, k, len(raw_filters or []))
        return []
    thresholder = get_threshold_postprocessor(similarity_threshold)
    if thresholder is not None:
        # Drop weakly-fused hits before the (expensive) cross-encoder sees them
        before_count = len(nodes)
        nodes = thresholder.postprocess_nodes(nodes)
        if len(nodes) < before_count:
            log.debug("threshold cut %d node(s)", before_count - len(nodes))
    if nodes and use_rerank:
        nodes = create_reranker(rerank_top_n, reranker_provider, reranker_model).postprocess_nodes(nodes, query_str=query)
    # use_rerank=False: eval harness measures the raw fused ranking without the cross-encoder
    log.info("retrieve: index=%s top_k=%d filters=%d -> %d node(s)", target_index, k, len(raw_filters or []), len(nodes))
    return nodes

def retrieve_debug(
    query: str,
    strategy: str | None = None,
    index_name: str | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    raw_filters: list[dict] | None = None,
    filter_condition: str = "and",
    similarity_threshold: float | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> dict[str, list[NodeWithScore]]:
    """Run BM25-only, dense-only, thresholded-hybrid and hybrid+rerank retrieval side by
    side, for UI/debug inspection. Returns a dict with keys "bm25", "dense", "hybrid",
    "reranked".

    Legs a backend does not support come back empty, so the UI can tell which are meaningful.
    The fused-score threshold is applied to the hybrid leg only, before rerank.
    """
    settings = get_settings()
    backend = create_backend(settings)
    target_index = build_index_name(strategy, index_name, settings)
    k = top_k or settings.retrieval_top_k
    settings = _resolve_embed_settings(settings, target_index, backend)

    def leg(mode: QueryMode) -> list[NodeWithScore]:
        # Empty when unsupported: no fabricated hits for modes the store can't serve.
        if mode not in backend.supported_query_modes:
            return []
        return _query_nodes(backend, target_index, query, mode, k, raw_filters, filter_condition, settings)

    try:
        bm25_nodes = leg(QueryMode.SPARSE)
        dense_nodes = leg(QueryMode.DENSE)
        hybrid_nodes = leg(QueryMode.HYBRID)
    except Exception as exc:
        raise VectorStoreError(f"Retrieval failed on index '{target_index}': {exc}") from exc

    thresholder = get_threshold_postprocessor(similarity_threshold)
    if thresholder is not None:
        # Threshold only the fused leg; BM25/dense legs keep raw scores for debugging
        hybrid_nodes = thresholder.postprocess_nodes(hybrid_nodes)

    # SentenceTransformerRerank assigns node.score in place, so hand it fresh NodeWithScore
    # wrappers: otherwise the "hybrid" leg we return below would carry the cross-encoder's
    # logits instead of the fused 0-1 scores it is documented to expose.
    rerank_input = [NodeWithScore(node=n.node, score=n.score) for n in hybrid_nodes]
    reranked_nodes = (
        create_reranker(rerank_top_n, reranker_provider, reranker_model).postprocess_nodes(rerank_input, query_str=query)
        if hybrid_nodes else []
    )
    return {"bm25": bm25_nodes, "dense": dense_nodes, "hybrid": hybrid_nodes, "reranked": reranked_nodes}