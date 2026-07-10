"""Tests for the provider-agnostic embedding factory; concrete providers live in src.providers."""
import pytest
from llama_index.core.embeddings import MockEmbedding
import src.providers.embeddings.fastembed as fe_mod
import src.providers.embeddings.watsonx as wx_mod
from src.factories import embeddings as emb_mod
from src.settings import Settings

class _StubEmbedding:
    """Captures constructor kwargs so tests never hit the real providers."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

def test_watsonx_provider(monkeypatch):
    monkeypatch.setattr(wx_mod, "WatsonxEmbeddings", _StubEmbedding)
    s = Settings(_env_file=None, embedding_provider="watsonx", watsonx_apikey="k", watsonx_project_id="p")
    model = emb_mod.create_embedding_model(s)
    assert model.kwargs["model_id"] == s.watsonx_embedding_model_id
    assert model.kwargs["url"] == s.watsonx_url
    assert model.kwargs["project_id"] == "p"

def test_fastembed_provider(monkeypatch):
    monkeypatch.setattr(fe_mod, "FastEmbedEmbedding", _StubEmbedding)
    s = Settings(_env_file=None, embedding_provider="fastembed")
    model = emb_mod.create_embedding_model(s)
    assert model.kwargs["model_name"] == s.fastembed_model_name

def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        emb_mod.create_embedding_model(Settings(_env_file=None, embedding_provider="banana"))

def test_custom_provider_registration():
    """Extensibility: a provider registered from anywhere is served by the factory."""
    marker = MockEmbedding(embed_dim=4)
    @emb_mod.EmbeddingFactory.register("fake-test-provider")
    def _build(settings):
        return marker
    s = Settings(_env_file=None, embedding_provider="fake-test-provider")
    assert emb_mod.create_embedding_model(s) is marker

def test_get_embedding_dim_probes_and_caches():
    emb_mod._DIM_CACHE.clear()
    mock = MockEmbedding(embed_dim=8)
    assert emb_mod.get_embedding_dim(mock) == 8
    # Second call must come from cache (same result, no re-probe needed to assert)
    assert emb_mod.get_embedding_dim(mock) == 8

def test_load_embedding_model_choices_reads_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "embedding_models.yaml"
    yaml_path.write_text("watsonx:\n  - model-a\nfastembed:\n  - model-b\n  - model-c\n")
    emb_mod.load_embedding_model_choices.cache_clear()
    monkeypatch.setattr(emb_mod, "EMBEDDING_MODELS_CONFIG_PATH", yaml_path)
    choices = emb_mod.load_embedding_model_choices()
    assert choices == {"watsonx": ["model-a"], "fastembed": ["model-b", "model-c"]}
    emb_mod.load_embedding_model_choices.cache_clear()


def test_get_embedding_dim_distinguishes_watsonx_model_ids():
    """Regression: watsonx models leave model_name unset, so the cache must key on model_id."""
    emb_mod._DIM_CACHE.clear()
    class _FakeWatsonx:
        """Mimics WatsonxEmbeddings: exposes model_id, model_name stays "unknown"."""
        model_name = "unknown"
        def __init__(self, model_id, dim):
            self.model_id = model_id
            self._dim = dim
        def get_text_embedding(self, text):
            return [0.0] * self._dim
    a = _FakeWatsonx("ibm/slate-125m-english-rtrvr", 768)
    b = _FakeWatsonx("ibm/granite-embedding-107m", 384)
    assert emb_mod.get_embedding_dim(a) == 768
    assert emb_mod.get_embedding_dim(b) == 384