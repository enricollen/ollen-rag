"""Shared test fixtures: isolate Settings from the developer's real .env."""
import pytest
from src.factories.embeddings import EmbeddingFactory
from src.factories.llm import LLMConnectorFactory
from src.factories.reranker import RerankerFactory
from src.factories.vector_store import VectorStoreFactory
from src.settings import get_settings

@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    """Force offline-safe defaults and clear the singleton before/after each test."""
    monkeypatch.setenv("OLLEN_RAG_WATSONX_APIKEY", "test-key")
    monkeypatch.setenv("OLLEN_RAG_WATSONX_PROJECT_ID", "test-project")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

@pytest.fixture(autouse=True)
def restore_factory_registries():
    """Snapshot/restore factory registries so tests registering fake providers don't leak
    into other tests (e.g. assertions on the 'Available providers' error message).

    EmbeddingFactory and RerankerFactory carry a second dict, _model_fields, which feeds
    default_models(); it must be restored alongside _registry or a fake provider outlives its test.
    """
    llm_snapshot = dict(LLMConnectorFactory._registry)
    vs_snapshot = dict(VectorStoreFactory._registry)
    emb_snapshot = dict(EmbeddingFactory._registry), dict(EmbeddingFactory._model_fields)
    rr_snapshot = dict(RerankerFactory._registry), dict(RerankerFactory._model_fields)
    yield
    LLMConnectorFactory._registry.clear()
    LLMConnectorFactory._registry.update(llm_snapshot)
    VectorStoreFactory._registry.clear()
    VectorStoreFactory._registry.update(vs_snapshot)
    for factory, (registry, model_fields) in ((EmbeddingFactory, emb_snapshot), (RerankerFactory, rr_snapshot)):
        factory._registry.clear()
        factory._registry.update(registry)
        factory._model_fields.clear()
        factory._model_fields.update(model_fields)