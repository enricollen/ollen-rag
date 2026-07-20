"""Tests for the fused-score similarity threshold (score floor before rerank)."""
from llama_index.core.schema import NodeWithScore, TextNode
from src.rag import retrieval as retrieval_mod
from src.settings import get_settings

def _node(node_id: str, score: float | None) -> NodeWithScore:
    """Build a scored node for threshold filtering tests."""
    return NodeWithScore(node=TextNode(id_=node_id, text="chunk"), score=score)

def test_threshold_disabled_by_default():
    # settings default similarity_threshold=0.0 -> feature off -> no postprocessor
    assert retrieval_mod.get_threshold_postprocessor(None) is None

def test_threshold_zero_explicit_disabled():
    assert retrieval_mod.get_threshold_postprocessor(0.0) is None

def test_threshold_from_settings(monkeypatch):
    monkeypatch.setenv("OLLEN_RAG_SIMILARITY_THRESHOLD", "0.25")
    get_settings.cache_clear()
    post = retrieval_mod.get_threshold_postprocessor(None)
    assert post is not None
    assert post.similarity_cutoff == 0.25

def test_threshold_override_beats_settings(monkeypatch):
    monkeypatch.setenv("OLLEN_RAG_SIMILARITY_THRESHOLD", "0.9")
    get_settings.cache_clear()
    post = retrieval_mod.get_threshold_postprocessor(0.1)
    assert post is not None
    assert post.similarity_cutoff == 0.1

def test_postprocessor_filters_below_cutoff_and_none_scores():
    post = retrieval_mod.get_threshold_postprocessor(0.5)
    kept = post.postprocess_nodes([_node("a", 0.9), _node("b", 0.5), _node("c", 0.49), _node("d", None)])
    assert [n.node.node_id for n in kept] == ["a", "b"]
