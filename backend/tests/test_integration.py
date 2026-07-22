"""End-to-end tests against a real OpenSearch (docker compose up opensearch).

Uses fastembed (CPU) so no watsonx credentials are needed.
Run explicitly: pytest -m integration
"""
import time
import httpx
import pytest
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from src.factories.chunker import create_node_parser
from src.factories.embeddings import create_embedding_model, get_embedding_dim
from src.factories.vector_store import create_backend
from src.rag.retrieval import retrieve
from src.settings import get_settings

# Mark every test in this module as integration-only so pyproject.toml's
# `addopts = -m "not integration"` excludes it from the default run.
pytestmark = pytest.mark.integration

INDEX = "ollen_rag_it_sentence"

def _opensearch_up(url: str) -> bool:
    # Skip the whole module when no local cluster is reachable
    try:
        return httpx.get(f"{url}/_cluster/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False

@pytest.fixture(scope="module", autouse=True)
def integration_env():
    """Force the fastembed provider for the whole module and skip cleanly if
    OpenSearch isn't reachable; cleans the test index before and after."""
    import os
    # Probe reachability BEFORE mutating the environment so the skip path
    # leaves no trace; try/finally guarantees cleanup even if a test errors.
    if not _opensearch_up(get_settings().opensearch_url):
        pytest.skip("OpenSearch not reachable; run: docker compose up -d opensearch")
    os.environ["OLLEN_RAG_EMBEDDING_PROVIDER"] = "fastembed"
    get_settings.cache_clear()
    settings = get_settings()
    try:
        # Clean the test index before and after the module
        httpx.delete(f"{settings.opensearch_url}/{INDEX}", timeout=10.0)
        yield settings
    finally:
        httpx.delete(f"{settings.opensearch_url}/{INDEX}", timeout=10.0)
        del os.environ["OLLEN_RAG_EMBEDDING_PROVIDER"]
        get_settings.cache_clear()

def test_full_ingest_and_hybrid_retrieve(integration_env):
    """Ingests two documents through the real pipeline and confirms hybrid
    retrieval returns the semantically relevant one first."""
    settings = integration_env
    backend = create_backend(settings)
    backend.warmup()
    embed_model = create_embedding_model(settings)
    documents = [
        Document(text="Il protocollo di triage del pronto soccorso assegna codici colore.", metadata={"team": "triage"}),
        Document(text="La pizza napoletana richiede farina, pomodoro e mozzarella.", metadata={"team": "cucina"}),
    ]
    backend.ensure_ready(INDEX, get_embedding_dim(embed_model))
    pipeline = IngestionPipeline(transformations=[create_node_parser("sentence", settings=settings), embed_model])
    nodes = pipeline.run(documents=documents)
    backend.add_nodes(INDEX, nodes)
    assert len(nodes) >= 2
    time.sleep(2)  # allow OpenSearch to refresh the index
    results = retrieve("codici colore triage", index_name=INDEX, top_k=2, rerank_top_n=1)
    assert results
    assert "triage" in results[0].node.get_content().lower()

def test_metadata_filter_restricts_results(integration_env):
    """Confirms metadata filters passed to retrieve() are honored, restricting
    results to nodes matching the given team filter."""
    results = retrieve(
        "pomodoro", index_name=INDEX, top_k=5, rerank_top_n=5,
        raw_filters=[{"key": "team", "value": "cucina"}],
    )
    assert results
    for node_with_score in results:
        assert node_with_score.node.metadata["team"] == "cucina"
