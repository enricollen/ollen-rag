"""Tests for the agnostic vector-store layer: index naming, registry, selection, mode fallback."""
import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from src.factories import vector_store as vs_mod
from src.factories.vector_store import (
    QueryMode, VectorStoreBackend, VectorStoreFactory,
    create_backend, embedding_meta, pick_supported_mode,
)
from src.settings import Settings

SETTINGS = Settings(_env_file=None)

# --- index naming (pure helper, stays in the agnostic layer) ---

def test_build_index_name_from_strategy():
    assert vs_mod.build_index_name("semantic", None, SETTINGS) == "ollen_rag_semantic"

def test_build_index_name_explicit_wins():
    assert vs_mod.build_index_name("semantic", "custom_idx", SETTINGS) == "custom_idx"

def test_build_index_name_default_strategy():
    assert vs_mod.build_index_name(None, None, SETTINGS) == "ollen_rag_sentence"

# --- registry / selection / capability ---

class FakeBackend(VectorStoreBackend):
    """Dense-only in-test backend recording the last retrieve() mode."""
    def __init__(self, settings=None):
        self.last_mode = None
        self._meta = {}

    @property
    def supported_query_modes(self):
        return {QueryMode.DENSE}

    def ensure_ready(self, index, dim):
        pass

    def add_nodes(self, index, nodes):
        pass

    def retrieve(self, index, query_str, query_embedding, mode, top_k, raw_filters, filter_condition):
        self.last_mode = mode
        return [NodeWithScore(node=TextNode(text="x"), score=1.0)]

    def get_index_meta(self, index):
        return self._meta.get(index)

    def set_index_meta(self, index, embedding_provider, embedding_model, chunking):
        self._meta[index] = {"embedding_provider": embedding_provider, "embedding_model": embedding_model, "chunking": chunking}

    def get_index_dim(self, index):
        return None

    def list_indices(self):
        return []

    def get_index_documents(self, index, offset, limit):
        return {"total": 0, "documents": []}

    def list_buckets(self, index):
        return []

    def list_bucket_files(self, index):
        return {}

    def find_duplicate_file(self, index, file_hash, bucket):
        return None

    def delete_index(self, index):
        pass

def test_create_backend_returns_registered_backend():
    VectorStoreFactory.register("fake")(FakeBackend)
    backend = create_backend(Settings(_env_file=None, vector_store="fake"))
    assert isinstance(backend, FakeBackend)

def test_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown vector store"):
        create_backend(Settings(_env_file=None, vector_store="nope"))

def test_pick_supported_mode_falls_back_to_dense():
    backend = FakeBackend()
    assert pick_supported_mode(backend, QueryMode.HYBRID) == QueryMode.DENSE

def test_pick_supported_mode_keeps_supported_mode():
    backend = FakeBackend()
    assert pick_supported_mode(backend, QueryMode.DENSE) == QueryMode.DENSE

def test_embedding_meta_extracts_embedding_view():
    backend = FakeBackend()
    backend.set_index_meta("idx", "watsonx", "slate", {"strategy": "sentence"})
    assert embedding_meta(backend, "idx") == {"embedding_provider": "watsonx", "embedding_model": "slate"}

def test_embedding_meta_none_when_missing():
    assert embedding_meta(FakeBackend(), "absent") is None