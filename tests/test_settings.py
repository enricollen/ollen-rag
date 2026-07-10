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
    # Multilingual by default: the English-only ms-marco cross-encoder measurably degrades
    # ranking on a non-English corpus (MRR 0.958 vs 1.00 with no reranking at all).
    assert s.reranker_model == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

def test_settings_env_override(monkeypatch):
    """OLLEN_RAG_ prefixed env vars must override defaults."""
    monkeypatch.setenv("OLLEN_RAG_EMBEDDING_PROVIDER", "fastembed")
    get_settings.cache_clear()
    assert get_settings().embedding_provider == "fastembed"
    get_settings.cache_clear()

def test_litellm_settings_defaults():
    """LiteLLM fields must all default so existing .env files keep working untouched."""
    s = Settings(_env_file=None)
    assert s.litellm_model == ""
    assert s.litellm_api_base == ""
    assert s.litellm_api_key == ""
    assert s.litellm_max_new_tokens == 800
    assert s.litellm_temperature == 0.1
    assert s.ollama_api_base == "http://localhost:11434"
    assert s.ollama_model == "llama3.1"

def test_litellm_settings_env_override(monkeypatch):
    """OLLEN_RAG_LITELLM_* env vars must override the defaults."""
    monkeypatch.setenv("OLLEN_RAG_LITELLM_MODEL", "openai/gpt-4o")
    get_settings.cache_clear()
    assert get_settings().litellm_model == "openai/gpt-4o"
    get_settings.cache_clear()