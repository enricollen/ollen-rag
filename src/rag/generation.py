"""Cited answer generation: hybrid retrieval + rerank + CitationQueryEngine + watsonx LLM."""
from llama_index.core.query_engine import CitationQueryEngine
from src.exceptions import GenerationError, OllenRagError
from src.factories.llm import create_llm
from src.factories.reranker import create_reranker
from src.factories.vector_store import build_index_name
from src.logger import OllenLogger
from src.prompts import load_prompt
from src.rag.retrieval import build_backend_retriever, get_threshold_postprocessor
from src.settings import get_settings

log = OllenLogger("generation")

def generate(
    query: str,
    strategy: str | None = None,
    index_name: str | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    raw_filters: list[dict] | None = None,
    filter_condition: str = "and",
    prompt_name: str | None = None,
    similarity_threshold: float | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> dict:
    """Answer a question with inline [n] citations; sources[] ids match the citation numbers."""
    settings = get_settings()
    target_index = build_index_name(strategy, index_name, settings)
    # Backend-driven retriever (works for any vector store); filters travel as raw dicts.
    retriever = build_backend_retriever(target_index, top_k or settings.retrieval_top_k, raw_filters, filter_condition)
    # CitationQueryEngine renumbers retrieved chunks as "Source n:" and prompts the LLM to cite them.
    # index=None is safe: from_args only uses index to derive a retriever, and we pass one explicitly.
    engine = CitationQueryEngine.from_args(
        None,
        retriever=retriever,
        llm=create_llm(settings),
        citation_qa_template=load_prompt(prompt_name or settings.default_prompt_name, settings),
        # Threshold (fused-score floor) runs before the reranker; None when disabled
        node_postprocessors=[
            p for p in (get_threshold_postprocessor(similarity_threshold), create_reranker(rerank_top_n, reranker_provider, reranker_model)) if p is not None
        ],
        citation_chunk_size=settings.citation_chunk_size,
    )
    try:
        response = engine.query(query)
    except OllenRagError:
        raise
    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc
    # source_nodes order matches the [n] numbering produced by the engine (1-based)
    sources = [
        {
            "id": position,
            "text": node_with_score.node.get_content(),
            # Rerank score: a 0-1 relevance probability, normalized by the connector (see
            # RerankConnector's contract). float() casts off numpy/torch scalar types so the API
            # layer can JSON-serialize it. A missing score stays None.
            "score": float(node_with_score.score) if node_with_score.score is not None else None,
            "metadata": node_with_score.node.metadata,
        }
        for position, node_with_score in enumerate(response.source_nodes, start=1)
    ]
    log.info("generate: index=%s prompt=%s -> %d source(s)", target_index, prompt_name or settings.default_prompt_name, len(sources))
    return {"answer": str(response), "sources": sources}
