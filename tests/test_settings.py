"""Tests for the central Settings object and its env override behaviour."""
from src.settings import Settings, get_settings

def test_settings_defaults():
    """Defaults must allow offline construction with no env vars set."""
    s = Settings(_env_file=None)
    assert s.embedding_provider == "watsonx"
    assert s.llm_provider == "watsonx"
    assert s.opensearch_index_prefix == "ollen_rag"
    assert s.default_chunking_strategy == "sentence"
    assert s.hybrid_dense_weight == 0.7

def test_settings_env_override(monkeypatch):
    """OLLEN_RAG_ prefixed env vars must override defaults."""
    monkeypatch.setenv("OLLEN_RAG_EMBEDDING_PROVIDER", "fastembed")
    get_settings.cache_clear()
    assert get_settings().embedding_provider == "fastembed"
    get_settings.cache_clear()