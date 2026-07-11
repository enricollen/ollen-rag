"""Tests for the LiteLLM-backed embedding providers. litellm.embedding is always monkeypatched:
these are unit tests and must never touch the network."""
import pytest
import src.providers.embeddings.litellm as lite_mod
from src.exceptions import EmbeddingError
from src.factories.embeddings import EmbeddingFactory, create_embedding_model
from src.settings import Settings

class _Recorder:
    """Stands in for litellm.embedding, capturing kwargs and returning a fixed vector."""
    def __init__(self):
        self.calls = []
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in kwargs["input"]]})()

@pytest.fixture
def recorder(monkeypatch):
    """Swap the litellm entrypoint for a recorder, per test."""
    rec = _Recorder()
    monkeypatch.setattr(lite_mod, "embedding", rec)
    return rec

def test_generic_provider_sends_the_verbatim_model_string(recorder):
    s = Settings(_env_file=None, embedding_provider="litellm",
                 litellm_embedding_model="openai/text-embedding-3-small", litellm_api_key="sk-x")
    model = create_embedding_model(s)
    assert model.get_text_embedding("hello") == [0.1, 0.2, 0.3]
    assert recorder.calls[0]["model"] == "openai/text-embedding-3-small"
    assert recorder.calls[0]["api_key"] == "sk-x"

def test_generic_provider_omits_empty_credentials(recorder):
    """LiteLLM treats api_key='' as a real (invalid) key rather than as absent."""
    s = Settings(_env_file=None, embedding_provider="litellm", litellm_embedding_model="ollama/x")
    create_embedding_model(s).get_text_embedding("hello")
    assert "api_key" not in recorder.calls[0]
    assert "api_base" not in recorder.calls[0]

def test_generic_provider_refuses_to_construct_without_a_model():
    s = Settings(_env_file=None, embedding_provider="litellm")
    with pytest.raises(ValueError, match="OLLEN_RAG_LITELLM_EMBEDDING_MODEL"):
        create_embedding_model(s)

def test_watsonx_provider_prefixes_the_model_and_sends_project_id(recorder):
    s = Settings(_env_file=None, embedding_provider="litellm-watsonx", watsonx_apikey="k",
                 watsonx_project_id="p", watsonx_embedding_model_id="ibm/slate-30m-english-rtrvr")
    create_embedding_model(s).get_text_embedding("hello")
    assert recorder.calls[0]["model"] == "watsonx/ibm/slate-30m-english-rtrvr"
    assert recorder.calls[0]["project_id"] == "p"
    assert recorder.calls[0]["api_key"] == "k"

def test_watsonx_provider_refuses_to_construct_without_a_project_id():
    # Explicit "" rather than an omitted field: conftest's clean_settings fixture exports
    # OLLEN_RAG_WATSONX_PROJECT_ID for every test, and env beats defaults but not init kwargs.
    s = Settings(_env_file=None, embedding_provider="litellm-watsonx", watsonx_apikey="k", watsonx_project_id="")
    with pytest.raises(ValueError, match="OLLEN_RAG_WATSONX_PROJECT_ID"):
        create_embedding_model(s)

def test_ollama_provider_needs_only_an_api_base(recorder):
    s = Settings(_env_file=None, embedding_provider="litellm-ollama", ollama_embedding_model="nomic-embed-text")
    create_embedding_model(s).get_text_embedding("hello")
    assert recorder.calls[0]["model"] == "ollama/nomic-embed-text"
    assert recorder.calls[0]["api_base"] == "http://localhost:11434"
    assert "api_key" not in recorder.calls[0]

def test_batch_embedding_sends_one_call(recorder):
    s = Settings(_env_file=None, embedding_provider="litellm-ollama")
    vectors = create_embedding_model(s)._get_text_embeddings(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["input"] == ["a", "b", "c"]

def test_provider_failure_becomes_embedding_error(monkeypatch):
    def _boom(**kwargs):
        """Simulate any vendor SDK raising from inside litellm."""
        raise RuntimeError("upstream 503")
    monkeypatch.setattr(lite_mod, "embedding", _boom)
    s = Settings(_env_file=None, embedding_provider="litellm-ollama")
    with pytest.raises(EmbeddingError, match="upstream 503"):
        create_embedding_model(s).get_text_embedding("hello")

def test_recorded_model_id_stays_bare_for_watsonx(recorder):
    """An index built with litellm-watsonx must record the same model id as native watsonx,
    so the two providers are interchangeable against one index."""
    s = Settings(_env_file=None, embedding_provider="litellm-watsonx", watsonx_project_id="p")
    assert EmbeddingFactory.resolve_model(s) == s.watsonx_embedding_model_id
    assert not EmbeddingFactory.resolve_model(s).startswith("watsonx/")
