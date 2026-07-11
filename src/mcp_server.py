"""FastMCP server exposing the RAG capabilities as MCP tools (mounted at /mcp by app.py)."""
from typing import Any
from fastmcp import FastMCP
from src.factories.vector_store import create_backend
from src.rag.generation import generate as generate_answer
from src.rag.ingestion import ingest_document as ingest_doc
from src.rag.retrieval import retrieve as retrieve_nodes

# Create the FastMCP server instance.
mcp = FastMCP("ollen-rag-service")


@mcp.tool
def ingest_document(
    file_path: str,
    strategy: str | None = None,
    index_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    enrich_keywords: bool | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    chunk_params: dict[str, Any] | None = None,
) -> dict:
    """Parse a server-local document (PDF/Office/images), chunk it with the given strategy, embed and store it in OpenSearch. Set enrich_keywords to add LLM-extracted search keywords to every chunk (slower; boosts keyword recall). embedding_provider/embedding_model pick the model for a NEW index (existing indices lock to their recorded model); chunk_params overrides the strategy's chunk knobs."""
    return ingest_doc(
        file_path, strategy=strategy, index_name=index_name, extra_metadata=metadata,
        enrich_keywords=enrich_keywords, embedding_provider=embedding_provider,
        embedding_model=embedding_model, chunk_params=chunk_params,
    )


@mcp.tool
def retrieve(
    query: str,
    strategy: str | None = None,
    index_name: str | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    filters: list[dict] | None = None,
    filter_condition: str = "and",
    similarity_threshold: float | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> dict:
    """Hybrid search (BM25 + dense) with optional metadata filters, reranked for relevance. similarity_threshold applies a fused-score floor before reranking; reranker_provider/reranker_model override the configured reranker (sentence-transformers | litellm | litellm-watsonx). Node scores are 0-1 relevance probabilities."""
    nodes = retrieve_nodes(
        query, strategy=strategy, index_name=index_name, top_k=top_k,
        rerank_top_n=rerank_top_n, raw_filters=filters, filter_condition=filter_condition,
        similarity_threshold=similarity_threshold, reranker_provider=reranker_provider,
        reranker_model=reranker_model,
    )
    return {
        "nodes": [
            {"text": n.node.get_content(), "score": n.score, "metadata": n.node.metadata}
            for n in nodes
        ]
    }


@mcp.tool
def rag_query(
    query: str,
    strategy: str | None = None,
    index_name: str | None = None,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    filters: list[dict] | None = None,
    filter_condition: str = "and",
    prompt_name: str | None = None,
    similarity_threshold: float | None = None,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> dict:
    """Answer a question using RAG: returns the cited answer plus numbered sources. similarity_threshold applies a fused-score floor before reranking; reranker_provider/reranker_model override the configured reranker (sentence-transformers | litellm | litellm-watsonx). Source scores are 0-1 relevance probabilities."""
    return generate_answer(
        query, strategy=strategy, index_name=index_name, top_k=top_k,
        rerank_top_n=rerank_top_n, raw_filters=filters,
        filter_condition=filter_condition, prompt_name=prompt_name,
        similarity_threshold=similarity_threshold, reranker_provider=reranker_provider,
        reranker_model=reranker_model,
    )


@mcp.tool
def list_indices() -> dict:
    """List the service-owned vector store indices (name + document count)."""
    return {"indices": create_backend().list_indices()}
