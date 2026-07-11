"""Tests for the LiteLLM-backed rerank connectors. litellm.rerank is always monkeypatched."""
import pytest
from llama_index.core.schema import NodeWithScore, TextNode
import src.providers.reranker.litellm as lite_mod
from src.exceptions import RerankError
from src.settings import Settings

class _Recorder:
    """Stands in for litellm.rerank, capturing kwargs and returning a Cohere-shaped response."""
    def __init__(self, results=None):
        self.calls = []
        self.results = results if results is not None else [
            {"index": 1, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.12},
        ]
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"results": self.results})()

@pytest.fixture
def recorder(monkeypatch):
    """Swap the litellm entrypoint for a recorder, per test."""
    rec = _Recorder()
    monkeypatch.setattr(lite_mod, "rerank", rec)
    return rec

def _nodes(*texts):
    """Fused-score nodes as they arrive from the vector store."""
    return [NodeWithScore(node=TextNode(text=t, id_=t), score=0.5) for t in texts]

def test_generic_connector_maps_results_back_to_nodes(recorder):
    """relevance_score is already 0-1; results carry the original node index."""
    s = Settings(_env_file=None, litellm_rerank_model="cohere/rerank-v3.5", litellm_api_key="sk-x")
    connector = lite_mod.LiteLLMRerankConnector(s)
    out = connector.rerank("q", _nodes("a", "b"), top_n=2)
    assert [n.node.id_ for n in out] == ["b", "a"]
    assert [n.score for n in out] == [0.91, 0.12]
    assert recorder.calls[0]["model"] == "cohere/rerank-v3.5"
    assert recorder.calls[0]["documents"] == ["a", "b"]
    assert recorder.calls[0]["top_n"] == 2
    assert recorder.calls[0]["api_key"] == "sk-x"

def test_generic_connector_omits_empty_credentials(recorder):
    """LiteLLM treats api_key='' as a real (invalid) key rather than as absent."""
    s = Settings(_env_file=None, litellm_rerank_model="cohere/rerank-v3.5")
    lite_mod.LiteLLMRerankConnector(s).rerank("q", _nodes("a", "b"), top_n=2)
    assert "api_key" not in recorder.calls[0]
    assert "api_base" not in recorder.calls[0]

def test_generic_connector_refuses_to_construct_without_a_model():
    with pytest.raises(ValueError, match="OLLEN_RAG_LITELLM_RERANK_MODEL"):
        lite_mod.LiteLLMRerankConnector(Settings(_env_file=None))

def test_watsonx_connector_prefixes_the_model_and_sends_project_id(recorder):
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    connector = lite_mod.LiteLLMWatsonxRerankConnector(s)
    connector.rerank("q", _nodes("a", "b"), top_n=2)
    assert recorder.calls[0]["model"] == "watsonx/cross-encoder/ms-marco-minilm-l-12-v2"
    assert recorder.calls[0]["project_id"] == "p"
    assert recorder.calls[0]["api_key"] == "k"
    assert recorder.calls[0]["api_base"] == s.watsonx_url

def test_watsonx_connector_sigmoids_its_raw_logits(monkeypatch):
    """watsonx returns the cross-encoder's raw logit as relevance_score, unlike Cohere/Jina which
    return a probability. These are the values a live call actually produced. Without the sigmoid
    the score would be 6.902, which is not a probability and breaks the connector contract."""
    monkeypatch.setattr(lite_mod, "rerank", _Recorder(results=[
        {"index": 1, "relevance_score": 6.90234375},
        {"index": 0, "relevance_score": -0.0004968643188476562},
    ]))
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    out = lite_mod.LiteLLMWatsonxRerankConnector(s).rerank("q", _nodes("a", "b"), top_n=2)
    assert [n.node.id_ for n in out] == ["b", "a"]
    assert out[0].score == pytest.approx(0.999, abs=1e-3)
    assert out[1].score == pytest.approx(0.5, abs=1e-3)
    assert all(0.0 <= n.score <= 1.0 for n in out)

def test_large_logits_do_not_overflow(monkeypatch):
    """to_probability is branch-wise for exactly this: exp(710) overflows a float."""
    monkeypatch.setattr(lite_mod, "rerank", _Recorder(results=[
        {"index": 0, "relevance_score": -800.0},
        {"index": 1, "relevance_score": 800.0},
    ]))
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    out = lite_mod.LiteLLMWatsonxRerankConnector(s).rerank("q", _nodes("a", "b"), top_n=2)
    assert [n.score for n in out] == [0.0, 1.0]

def test_watsonx_connector_refuses_to_construct_without_a_project_id():
    # Explicit "" rather than an omitted field: conftest's clean_settings fixture exports
    # OLLEN_RAG_WATSONX_PROJECT_ID for every test, and env beats defaults but not init kwargs.
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="")
    with pytest.raises(ValueError, match="OLLEN_RAG_WATSONX_PROJECT_ID"):
        lite_mod.LiteLLMWatsonxRerankConnector(s)

def test_generic_connector_passes_probabilities_through_unchanged(recorder):
    """Cohere-style vendors already return 0-1; a sigmoid here would squash 0.91 into 0.713."""
    s = Settings(_env_file=None, litellm_rerank_model="cohere/rerank-v3.5")
    out = lite_mod.LiteLLMRerankConnector(s).rerank("q", _nodes("a", "b"), top_n=2)
    assert all(0.0 <= n.score <= 1.0 for n in out)
    assert out[0].score == 0.91
    assert out[1].score == 0.12

def test_index_zero_is_not_treated_as_missing(monkeypatch):
    """RerankResponseResult is a TypedDict; index 0 is falsy and must survive lookup."""
    monkeypatch.setattr(lite_mod, "rerank", _Recorder(results=[{"index": 0, "relevance_score": 0.8}]))
    s = Settings(_env_file=None, litellm_rerank_model="cohere/rerank-v3.5")
    out = lite_mod.LiteLLMRerankConnector(s).rerank("q", _nodes("a", "b"), top_n=1)
    assert out[0].node.id_ == "a"

def test_provider_failure_becomes_rerank_error(monkeypatch):
    def _boom(**kwargs):
        """Simulate any vendor SDK raising from inside litellm."""
        raise RuntimeError("upstream 503")
    monkeypatch.setattr(lite_mod, "rerank", _boom)
    s = Settings(_env_file=None, litellm_rerank_model="cohere/rerank-v3.5")
    with pytest.raises(RerankError, match="upstream 503"):
        lite_mod.LiteLLMRerankConnector(s).rerank("q", _nodes("a"), top_n=1)
