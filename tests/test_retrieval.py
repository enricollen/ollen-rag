"""Tests for backend-driven hybrid retrieval, embed-model resolution, and reranking wiring."""
import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from src.exceptions import VectorStoreError
from src.factories.vector_store import QueryMode
from src.rag import retrieval

def _node(text, score):
    return NodeWithScore(node=TextNode(text=text), score=score)

class _StubEmbed:
    """Minimal embed model: a fixed query vector, no model load."""
    def get_query_embedding(self, query):
        return [0.1, 0.2, 0.3]

class _FakeBackend:
    """Records retrieve() calls and returns canned nodes (per-mode or single list)."""
    def __init__(self, nodes=None, per_mode=None, error=None, meta=None, modes=None):
        self._nodes = nodes or []
        self._per_mode = per_mode
        self._error = error
        self._meta = meta
        self._modes = modes or {QueryMode.DENSE, QueryMode.SPARSE, QueryMode.HYBRID}
        self.calls = []

    @property
    def supported_query_modes(self):
        return self._modes

    def get_index_meta(self, index):
        return self._meta

    def retrieve(self, index, query_str, query_embedding, mode, top_k, raw_filters, filter_condition):
        self.calls.append(mode)
        if self._error:
            raise self._error
        if self._per_mode is not None:
            return self._per_mode.get(mode, [])
        return self._nodes

class _FakeReranker:
    """Orders nodes by score desc and truncates to top_n, like the real cross-encoder postprocessor."""
    def __init__(self, top_n):
        self.top_n = top_n
    def postprocess_nodes(self, nodes, query_str=None):
        return sorted(nodes, key=lambda n: n.score, reverse=True)[: self.top_n]

class _FakeSentenceTransformerRerank:
    """Stands in for llama_index's SentenceTransformerRerank: records constructor args, no model load."""
    def __init__(self, model, top_n):
        self.model = model
        self.top_n = top_n

def _patch_embed(monkeypatch):
    monkeypatch.setattr(retrieval, "create_embedding_model", lambda settings=None: _StubEmbed())

# --- retrieve ---

def test_retrieve_reranks(monkeypatch):
    nodes = [_node("low", 0.1), _node("high", 0.9), _node("mid", 0.5)]
    backend = _FakeBackend(nodes=nodes)
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: backend)
    monkeypatch.setattr(retrieval, "get_reranker", lambda top_n=None, model=None: _FakeReranker(top_n or 2))
    _patch_embed(monkeypatch)
    result = retrieval.retrieve("question?", rerank_top_n=2)
    assert [n.node.text for n in result] == ["high", "mid"]
    assert backend.calls == [QueryMode.HYBRID]  # main path requests hybrid

def test_retrieve_forwards_reranker_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(nodes=[_node("only", 0.5)]))
    def fake_get_reranker(top_n=None, model=None):
        captured["model"] = model
        return _FakeReranker(top_n or 2)
    monkeypatch.setattr(retrieval, "get_reranker", fake_get_reranker)
    _patch_embed(monkeypatch)
    retrieval.retrieve("q?", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert captured["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

def test_retrieve_wraps_store_errors(monkeypatch):
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(error=RuntimeError("os down")))
    _patch_embed(monkeypatch)
    with pytest.raises(VectorStoreError):
        retrieval.retrieve("question?")

def test_retrieve_empty_short_circuits(monkeypatch):
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(nodes=[]))
    _patch_embed(monkeypatch)
    called = {"rerank": False}
    monkeypatch.setattr(retrieval, "get_reranker", lambda *a, **k: called.__setitem__("rerank", True))
    assert retrieval.retrieve("q?") == []
    assert called["rerank"] is False  # no rerank on empty result

# --- retrieve_debug ---

def test_retrieve_debug_returns_all_legs(monkeypatch):
    per_mode = {
        QueryMode.SPARSE: [_node("bm25-a", 3.0), _node("bm25-b", 1.0)],
        QueryMode.DENSE: [_node("dense-a", 0.9), _node("dense-b", 0.4)],
        QueryMode.HYBRID: [_node("low", 0.1), _node("high", 0.9)],
    }
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(per_mode=per_mode))
    monkeypatch.setattr(retrieval, "get_reranker", lambda top_n=None, model=None: _FakeReranker(top_n or 2))
    _patch_embed(monkeypatch)
    result = retrieval.retrieve_debug("question?", rerank_top_n=2)
    assert set(result) == {"bm25", "dense", "hybrid", "reranked"}
    assert [n.node.text for n in result["bm25"]] == ["bm25-a", "bm25-b"]
    assert [n.node.text for n in result["dense"]] == ["dense-a", "dense-b"]
    assert [n.node.text for n in result["reranked"]] == ["high", "low"]
    assert result["hybrid"] == per_mode[QueryMode.HYBRID]

def test_retrieve_debug_empties_unsupported_legs(monkeypatch):
    """A dense-only backend must serve dense but leave bm25/hybrid empty."""
    per_mode = {QueryMode.DENSE: [_node("dense-a", 0.9)]}
    backend = _FakeBackend(per_mode=per_mode, modes={QueryMode.DENSE})
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: backend)
    monkeypatch.setattr(retrieval, "get_reranker", lambda top_n=None, model=None: _FakeReranker(top_n or 2))
    _patch_embed(monkeypatch)
    result = retrieval.retrieve_debug("q?")
    assert result["bm25"] == [] and result["hybrid"] == []
    assert [n.node.text for n in result["dense"]] == ["dense-a"]
    assert backend.calls == [QueryMode.DENSE]  # unsupported legs never hit the backend

def test_retrieve_debug_forwards_reranker_model(monkeypatch):
    per_mode = {QueryMode.HYBRID: [_node("only", 0.5)]}
    captured = {}
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(per_mode=per_mode))
    def fake_get_reranker(top_n=None, model=None):
        captured["model"] = model
        return _FakeReranker(top_n or 2)
    monkeypatch.setattr(retrieval, "get_reranker", fake_get_reranker)
    _patch_embed(monkeypatch)
    retrieval.retrieve_debug("q?", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert captured["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- embed-model resolution ---

def test_resolve_embed_uses_index_recorded_model(monkeypatch):
    captured = {}
    backend = _FakeBackend(nodes=[_node("only", 0.5)],
                           meta={"embedding_provider": "fastembed", "embedding_model": "BAAI/bge-large-en-v1.5"})
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: backend)
    def fake_create_embedding_model(settings=None):
        captured["provider"] = settings.embedding_provider
        captured["fastembed_model_name"] = settings.fastembed_model_name
        return _StubEmbed()
    monkeypatch.setattr(retrieval, "create_embedding_model", fake_create_embedding_model)
    monkeypatch.setattr(retrieval, "get_reranker", lambda top_n=None, model=None: _FakeReranker(2))
    retrieval.retrieve("q?", index_name="ollen_rag_sentence")
    assert captured["provider"] == "fastembed"
    assert captured["fastembed_model_name"] == "BAAI/bge-large-en-v1.5"

def test_resolve_embed_falls_back_when_no_meta(monkeypatch):
    captured = {}
    backend = _FakeBackend(nodes=[_node("only", 0.5)], meta=None)
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: backend)
    def fake_create_embedding_model(settings=None):
        captured["provider"] = settings.embedding_provider
        return _StubEmbed()
    monkeypatch.setattr(retrieval, "create_embedding_model", fake_create_embedding_model)
    monkeypatch.setattr(retrieval, "get_reranker", lambda top_n=None, model=None: _FakeReranker(2))
    retrieval.retrieve("q?", index_name="ollen_rag_sentence")
    assert captured["provider"] == retrieval.get_settings().embedding_provider  # unchanged global default

def test_build_backend_retriever_runs_hybrid_query(monkeypatch):
    backend = _FakeBackend(nodes=[_node("hit", 0.5)])
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: backend)
    _patch_embed(monkeypatch)
    from llama_index.core.schema import QueryBundle
    r = retrieval.build_backend_retriever("ollen_rag_sentence", 10)
    nodes = r._retrieve(QueryBundle(query_str="q?"))
    assert [n.node.text for n in nodes] == ["hit"]
    assert backend.calls == [QueryMode.HYBRID]

# --- reranker helpers (unchanged behavior) ---

def test_load_reranker_model_choices_reads_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "reranker_models.yaml"
    yaml_path.write_text("default: models/reranker\nalt: cross-encoder/ms-marco-MiniLM-L-6-v2\n")
    retrieval.load_reranker_model_choices.cache_clear()
    monkeypatch.setattr(retrieval, "RERANKER_MODELS_CONFIG_PATH", yaml_path)
    choices = retrieval.load_reranker_model_choices()
    assert choices == {"default": "models/reranker", "alt": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    retrieval.load_reranker_model_choices.cache_clear()

def test_get_reranker_caches_per_model(monkeypatch):
    retrieval._rerankers.clear()
    monkeypatch.setattr(retrieval, "SentenceTransformerRerank", _FakeSentenceTransformerRerank)
    monkeypatch.setattr(retrieval, "load_reranker_model_choices", lambda: {"default": "models/reranker", "alt": "cross-encoder/ms-marco-MiniLM-L-6-v2"})
    a = retrieval.get_reranker(top_n=3, model="models/reranker")
    b = retrieval.get_reranker(top_n=5, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    a_again = retrieval.get_reranker(top_n=7, model="models/reranker")
    assert a.model == "models/reranker"
    assert b.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert a_again is a  # cached, same instance
    assert a_again.top_n == 7  # top_n still updates on cache hit
    retrieval._rerankers.clear()

def test_get_reranker_none_model_uses_settings_default(monkeypatch):
    retrieval._rerankers.clear()
    monkeypatch.setattr(retrieval, "SentenceTransformerRerank", _FakeSentenceTransformerRerank)
    reranker = retrieval.get_reranker()
    assert reranker.model == retrieval.get_settings().reranker_model
    retrieval._rerankers.clear()

def test_get_reranker_rejects_unknown_model(monkeypatch):
    retrieval._rerankers.clear()
    monkeypatch.setattr(retrieval, "load_reranker_model_choices", lambda: {"default": "models/reranker"})
    with pytest.raises(ValueError):
        retrieval.get_reranker(model="banana")
    retrieval._rerankers.clear()