"""Tests for backend-driven hybrid retrieval, embed-model resolution, and reranking wiring.

The reranker itself is covered by test_reranker_factory.py and test_reranker_sentence_transformers.py;
here create_reranker is always patched out, so no cross-encoder is ever loaded.
"""
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

class _MutatingReranker:
    """Faithful stand-in for llama_index's SentenceTransformerRerank, which assigns
    `node.score = score` on the nodes it is handed (mutating the caller's list in place)
    and returns cross-encoder logits, not fused 0-1 scores."""
    def __init__(self, top_n):
        self.top_n = top_n
    def postprocess_nodes(self, nodes, query_str=None):
        for offset, node in enumerate(nodes):
            node.score = -10.0 - offset
        return sorted(nodes, key=lambda n: n.score, reverse=True)[: self.top_n]

def _patch_embed(monkeypatch):
    monkeypatch.setattr(retrieval, "create_embedding_model", lambda settings=None: _StubEmbed())

# --- retrieve ---

def test_retrieve_reranks(monkeypatch):
    nodes = [_node("low", 0.1), _node("high", 0.9), _node("mid", 0.5)]
    backend = _FakeBackend(nodes=nodes)
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: backend)
    monkeypatch.setattr(retrieval, "create_reranker", lambda top_n=None, provider=None, model=None: _FakeReranker(top_n or 2))
    _patch_embed(monkeypatch)
    result = retrieval.retrieve("question?", rerank_top_n=2)
    assert [n.node.text for n in result] == ["high", "mid"]
    assert backend.calls == [QueryMode.HYBRID]  # main path requests hybrid

def test_retrieve_forwards_reranker_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(nodes=[_node("only", 0.5)]))
    def fake_create_reranker(top_n=None, provider=None, model=None):
        captured["model"] = model
        return _FakeReranker(top_n or 2)
    monkeypatch.setattr(retrieval, "create_reranker", fake_create_reranker)
    _patch_embed(monkeypatch)
    retrieval.retrieve("q?", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert captured["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

def test_retrieve_forwards_reranker_provider(monkeypatch):
    """Per-request provider override reaches the factory alongside the model."""
    captured = {}
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(nodes=[_node("only", 0.5)]))
    def fake_create_reranker(top_n=None, provider=None, model=None):
        captured["provider"] = provider
        return _FakeReranker(top_n or 2)
    monkeypatch.setattr(retrieval, "create_reranker", fake_create_reranker)
    _patch_embed(monkeypatch)
    retrieval.retrieve("q?", reranker_provider="litellm-watsonx")
    assert captured["provider"] == "litellm-watsonx"

def test_retrieve_wraps_store_errors(monkeypatch):
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(error=RuntimeError("os down")))
    _patch_embed(monkeypatch)
    with pytest.raises(VectorStoreError):
        retrieval.retrieve("question?")

def test_retrieve_empty_short_circuits(monkeypatch):
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(nodes=[]))
    _patch_embed(monkeypatch)
    called = {"rerank": False}
    monkeypatch.setattr(retrieval, "create_reranker", lambda *a, **k: called.__setitem__("rerank", True))
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
    monkeypatch.setattr(retrieval, "create_reranker", lambda top_n=None, provider=None, model=None: _FakeReranker(top_n or 2))
    _patch_embed(monkeypatch)
    result = retrieval.retrieve_debug("question?", rerank_top_n=2)
    assert set(result) == {"bm25", "dense", "hybrid", "reranked"}
    assert [n.node.text for n in result["bm25"]] == ["bm25-a", "bm25-b"]
    assert [n.node.text for n in result["dense"]] == ["dense-a", "dense-b"]
    assert [n.node.text for n in result["reranked"]] == ["high", "low"]
    assert result["hybrid"] == per_mode[QueryMode.HYBRID]

def test_retrieve_debug_hybrid_leg_keeps_fused_scores(monkeypatch):
    """The reranker overwrites NodeWithScore.score in place, so retrieve_debug must not feed it
    the same objects it returns as the pre-rerank "hybrid" leg: consumers would then read
    cross-encoder logits (negative, unbounded) where a fused 0-1 score is documented."""
    per_mode = {QueryMode.HYBRID: [_node("high", 1.0), _node("low", 0.25)]}
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(per_mode=per_mode))
    monkeypatch.setattr(retrieval, "create_reranker", lambda top_n=None, provider=None, model=None: _MutatingReranker(top_n or 2))
    _patch_embed(monkeypatch)
    result = retrieval.retrieve_debug("q?", rerank_top_n=2)
    assert [n.score for n in result["hybrid"]] == [1.0, 0.25]
    # The reranked leg still carries the cross-encoder's own scores
    assert [n.score for n in result["reranked"]] == [-10.0, -11.0]

def test_retrieve_debug_empties_unsupported_legs(monkeypatch):
    """A dense-only backend must serve dense but leave bm25/hybrid empty."""
    per_mode = {QueryMode.DENSE: [_node("dense-a", 0.9)]}
    backend = _FakeBackend(per_mode=per_mode, modes={QueryMode.DENSE})
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: backend)
    monkeypatch.setattr(retrieval, "create_reranker", lambda top_n=None, provider=None, model=None: _FakeReranker(top_n or 2))
    _patch_embed(monkeypatch)
    result = retrieval.retrieve_debug("q?")
    assert result["bm25"] == [] and result["hybrid"] == []
    assert [n.node.text for n in result["dense"]] == ["dense-a"]
    assert backend.calls == [QueryMode.DENSE]  # unsupported legs never hit the backend

def test_retrieve_debug_forwards_reranker_model(monkeypatch):
    per_mode = {QueryMode.HYBRID: [_node("only", 0.5)]}
    captured = {}
    monkeypatch.setattr(retrieval, "create_backend", lambda settings=None: _FakeBackend(per_mode=per_mode))
    def fake_create_reranker(top_n=None, provider=None, model=None):
        captured["model"] = model
        return _FakeReranker(top_n or 2)
    monkeypatch.setattr(retrieval, "create_reranker", fake_create_reranker)
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
    monkeypatch.setattr(retrieval, "create_reranker", lambda top_n=None, provider=None, model=None: _FakeReranker(2))
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
    monkeypatch.setattr(retrieval, "create_reranker", lambda top_n=None, provider=None, model=None: _FakeReranker(2))
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
