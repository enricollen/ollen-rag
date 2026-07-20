"""Tests for the curated provider -> [model id] catalogs, the shared request validator, and the
provider -> Settings-field resolution that EmbeddingFactory and RerankerFactory delegate here."""
import pytest
from src.factories.model_catalog import (
    EMBEDDING_MODELS_CONFIG_PATH, RERANKER_MODELS_CONFIG_PATH, default_models, load_model_choices,
    model_field, resolve_model, validate_model, with_model,
)
from src.settings import Settings

CHOICES = {"watsonx": ["ibm/slate-125m-english-rtrvr"], "litellm": []}

# The mapping a factory collects at registration time: provider -> Settings field holding its model.
FIELDS = {"watsonx": "watsonx_embedding_model_id", "fastembed": "fastembed_model_name"}

def test_known_model_passes():
    validate_model(CHOICES, "watsonx", "ibm/slate-125m-english-rtrvr")

def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider 'banana'"):
        validate_model(CHOICES, "banana", None)

def test_unknown_model_for_curated_provider_raises():
    with pytest.raises(ValueError, match="Unknown model 'gpt-9'"):
        validate_model(CHOICES, "watsonx", "gpt-9")

def test_empty_list_means_free_form():
    """The generic litellm provider reaches a new vendor with no code or config change."""
    validate_model(CHOICES, "litellm", "cohere/embed-english-v3.0")

def test_none_model_always_passes():
    """No override requested: the provider's configured default applies."""
    validate_model(CHOICES, "watsonx", None)

def test_embedding_yaml_loads_with_litellm_providers():
    choices = load_model_choices(EMBEDDING_MODELS_CONFIG_PATH)
    assert choices["litellm"] == []
    assert "nomic-embed-text" in choices["litellm-ollama"]
    assert "ibm/slate-125m-english-rtrvr" in choices["litellm-watsonx"]

def test_reranker_yaml_is_provider_keyed():
    """Reshaped from the old flat label -> id map; the 'default:' pseudo-label is gone."""
    choices = load_model_choices(RERANKER_MODELS_CONFIG_PATH)
    assert "default" not in choices
    assert choices["litellm"] == []
    assert "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" in choices["sentence-transformers"]

# --- provider -> Settings field resolution ---
#
# These pure functions replace the six hand-written copies of this mapping that used to live in
# ingestion, retrieval, and two UI pages. The dict-literal copies raised KeyError on an unknown
# provider; the `if provider == "watsonx" else ...` copies silently reported the wrong model.

def test_model_field_lookup():
    assert model_field(FIELDS, "fastembed") == "fastembed_model_name"

def test_model_field_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider 'banana'"):
        model_field(FIELDS, "banana")

def test_resolve_model_reads_the_providers_own_field():
    """The active provider decides which field holds 'the' model."""
    s = Settings(_env_file=None, embedding_provider="fastembed", fastembed_model_name="bge")
    assert resolve_model(s, "embedding_provider", FIELDS) == "bge"
    assert resolve_model(s, "embedding_provider", FIELDS, provider="watsonx") == s.watsonx_embedding_model_id

def test_with_model_round_trips():
    """with_model writes into the provider's field; resolve_model reads it back."""
    s = Settings(_env_file=None, embedding_provider="watsonx")
    pinned = with_model(s, "embedding_provider", FIELDS, "fastembed", "intfloat/multilingual-e5-large")
    assert pinned.embedding_provider == "fastembed"
    assert pinned.fastembed_model_name == "intfloat/multilingual-e5-large"
    assert resolve_model(pinned, "embedding_provider", FIELDS) == "intfloat/multilingual-e5-large"

def test_with_model_without_model_only_switches_provider():
    s = Settings(_env_file=None, embedding_provider="watsonx", fastembed_model_name="kept")
    pinned = with_model(s, "embedding_provider", FIELDS, "fastembed", None)
    assert pinned.fastembed_model_name == "kept"

def test_with_model_rejects_unknown_provider():
    """A wrong provider name must raise, not silently fall through to some other field."""
    s = Settings(_env_file=None)
    with pytest.raises(ValueError, match="Unknown provider 'banana'"):
        with_model(s, "embedding_provider", FIELDS, "banana", "some-model")

def test_default_models_maps_every_provider():
    s = Settings(_env_file=None)
    assert default_models(s, FIELDS) == {
        "watsonx": s.watsonx_embedding_model_id,
        "fastembed": s.fastembed_model_name,
    }
