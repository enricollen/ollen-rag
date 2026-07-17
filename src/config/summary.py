"""Single source of truth for the active provider + model for each
component (LLM / embedding / reranker) plus the vector store and chunking. Both the startup log
(app.py) and the UI's active-config banner (/api/v1/config) read it, so they never drift."""
from src.factories.embeddings import EmbeddingFactory
from src.factories.reranker import RerankerFactory
from src.settings import Settings

def _llm_model(s: Settings) -> str:
    """Active LLM model id. Unlike embeddings/reranker there is no model-field registry to
    delegate to, so the provider -> field mapping lives here."""
    return {
        "watsonx": s.watsonx_llm_model_id, "litellm-watsonx": s.watsonx_llm_model_id,
        "litellm": s.litellm_model, "litellm-ollama": s.ollama_model,
        "litellm-openai": s.openai_model, "litellm-openrouter": s.openrouter_model,
    }.get(s.llm_provider, "?")

def component_summary(s: Settings) -> dict:
    """Resolved active configuration, provider + model per component. Registries must be populated
    (import src.providers) before calling so the factory model lookups see every provider."""
    try:
        embedding_model = EmbeddingFactory.resolve_model(s)
    except Exception:
        embedding_model = "?"
    try:
        reranker_model = RerankerFactory.resolve_model(s)
    except Exception:
        reranker_model = "?"
    return {
        "llm": {"provider": s.llm_provider, "model": _llm_model(s)},
        "embedding": {"provider": s.embedding_provider, "model": embedding_model},
        "reranker": {"provider": s.reranker_provider, "model": reranker_model},
        "vector_store": s.vector_store,
        "chunking": {
            "strategy": s.default_chunking_strategy,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
        },
        "retrieval_top_k": s.retrieval_top_k,
        "rerank_top_n": s.rerank_top_n,
    }
