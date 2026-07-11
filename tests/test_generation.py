"""Tests for cited generation via CitationQueryEngine with a MockLLM (offline)."""
import pytest
from llama_index.core import PromptTemplate
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode
from src.exceptions import GenerationError
from src.rag import generation

class _FakeRetriever:
    """Feeds two fixed source nodes into the citation engine."""
    def __init__(self, error=None):
        self.error = error
    def retrieve(self, query):
        if self.error:
            raise self.error
        return [
            NodeWithScore(node=TextNode(text="Il gatto dorme sul divano.", metadata={"file_name": "a.pdf"}), score=0.9),
            NodeWithScore(node=TextNode(text="Il cane gioca in giardino.", metadata={"file_name": "b.pdf"}), score=0.7),
        ]

class _PassThroughReranker:
    """No-op postprocessor standing in for the cross-encoder.

    Extended beyond the brief's minimal signature: the installed llama-index
    CitationQueryEngine calls `postprocess_nodes(nodes, query_bundle=...)`
    (not `query_str=...`), so `query_bundle` is accepted here too.
    """
    top_n = 4
    def postprocess_nodes(self, nodes, query_str=None, query_bundle=None):
        return nodes
    # CitationQueryEngine calls postprocessors via _apply_node_postprocessors; keep interface minimal
    callback_manager = None

@pytest.fixture
def mocked_generation(monkeypatch):
    monkeypatch.setattr(generation, "build_backend_retriever", lambda idx, k, raw_filters=None, filter_condition="and": _FakeRetriever())
    monkeypatch.setattr(generation, "create_reranker", lambda top_n=None, provider=None, model=None: _PassThroughReranker())
    monkeypatch.setattr(generation, "create_llm", lambda settings=None: MockLLM(max_tokens=64))
    monkeypatch.setattr(
        generation, "load_prompt",
        lambda name, settings=None: PromptTemplate("Fonti:\n{context_str}\nDomanda: {query_str}\nRisposta:"),
    )

def test_generate_returns_answer_and_numbered_sources(mocked_generation):
    result = generation.generate("Dove dorme il gatto?")
    assert isinstance(result["answer"], str) and result["answer"]
    assert len(result["sources"]) >= 1
    # Citation ids are 1-based and aligned with source order
    assert result["sources"][0]["id"] == 1
    assert "text" in result["sources"][0]
    assert "metadata" in result["sources"][0]

def test_generate_wraps_errors(monkeypatch, mocked_generation):
    monkeypatch.setattr(generation, "build_backend_retriever", lambda idx, k, raw_filters=None, filter_condition="and": _FakeRetriever(error=RuntimeError("boom")))
    with pytest.raises(GenerationError):
        generation.generate("Dove dorme il gatto?")

def test_generate_forwards_reranker_model(monkeypatch, mocked_generation):
    captured = {}
    def fake_create_reranker(top_n=None, provider=None, model=None):
        captured["model"] = model
        return _PassThroughReranker()
    monkeypatch.setattr(generation, "create_reranker", fake_create_reranker)
    generation.generate("Dove dorme il gatto?", reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert captured["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_generate_source_score_passes_through_the_connectors_probability(monkeypatch, mocked_generation):
    """sources[].score is whatever the reranker connector produced, unchanged.

    Normalization is the connector's job now (RerankConnector's contract guarantees a 0-1
    probability), so generation must not apply a second sigmoid. It used to: _relevance() lived
    here and squashed the score, which would have double-sigmoided LiteLLM's already-calibrated
    relevance_score."""
    class _ProbabilityRetriever:
        def retrieve(self, query):
            """A node scored by a connector honoring the 0-1 contract."""
            return [NodeWithScore(node=TextNode(text="Il gatto dorme.", metadata={}), score=0.00473)]
    monkeypatch.setattr(generation, "build_backend_retriever", lambda idx, k, raw_filters=None, filter_condition="and": _ProbabilityRetriever())
    result = generation.generate("Dove dorme il gatto?")
    assert result["sources"][0]["score"] == pytest.approx(0.00473)
    assert 0.0 < result["sources"][0]["score"] < 1.0

def test_generate_source_score_none_stays_none(monkeypatch, mocked_generation):
    """A node with no score must not become sigmoid(0)=0.5, which would invent a relevance signal."""
    class _NoScoreRetriever:
        def retrieve(self, query):
            return [NodeWithScore(node=TextNode(text="Il gatto dorme.", metadata={}), score=None)]
    monkeypatch.setattr(generation, "build_backend_retriever", lambda idx, k, raw_filters=None, filter_condition="and": _NoScoreRetriever())
    result = generation.generate("Dove dorme il gatto?")
    assert result["sources"][0]["score"] is None

def test_generate_source_score_is_plain_float(monkeypatch, mocked_generation):
    # SentenceTransformerRerank assigns numpy.float32 scores; generate() must
    # coerce them to plain float so the API layer can JSON-serialize sources.
    import numpy
    class _NumpyScoreRetriever:
        def retrieve(self, query):
            return [NodeWithScore(node=TextNode(text="Il gatto dorme.", metadata={}), score=numpy.float32(0.5))]
    monkeypatch.setattr(generation, "build_backend_retriever", lambda idx, k, raw_filters=None, filter_condition="and": _NumpyScoreRetriever())
    result = generation.generate("Dove dorme il gatto?")
    assert type(result["sources"][0]["score"]) is float
