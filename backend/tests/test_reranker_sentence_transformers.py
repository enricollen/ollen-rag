"""The local cross-encoder connector. CrossEncoder is monkeypatched: no model download in unit tests."""
import pytest
from llama_index.core.schema import NodeWithScore, TextNode
import src.providers.reranker.sentence_transformers as st_mod
from src.settings import Settings

class _StubCrossEncoder:
    """Emits fixed logits, applies the requested activation, and records which one it got.

    Applying activation_fn is what the real CrossEncoder.predict() does; a stub that ignored it
    would pass even if the connector forgot to ask for a sigmoid.
    """
    last_activation = None
    LOGITS = [-2.0, 3.0]
    def __init__(self, model_id):
        self.model_id = model_id
    def predict(self, pairs, activation_fn=None):
        """Stand in for the real cross-encoder over (query, passage) pairs."""
        import torch
        type(self).last_activation = activation_fn
        logits = torch.tensor(self.LOGITS[: len(pairs)])
        return activation_fn(logits).tolist() if activation_fn is not None else logits.tolist()

@pytest.fixture
def stub(monkeypatch):
    """Swap CrossEncoder and drop the lru_cache holding any previously built one."""
    st_mod._cross_encoder.cache_clear()
    monkeypatch.setattr(st_mod, "CrossEncoder", _StubCrossEncoder)
    yield _StubCrossEncoder
    st_mod._cross_encoder.cache_clear()

def test_scores_are_probabilities_sorted_descending(stub):
    """The contract: 0-1 relevance, best first. sigmoid(3.0)=0.953, sigmoid(-2.0)=0.119."""
    connector = st_mod.SentenceTransformerRerankConnector(Settings(_env_file=None))
    nodes = [NodeWithScore(node=TextNode(text="a", id_="a")), NodeWithScore(node=TextNode(text="b", id_="b"))]
    out = connector.rerank("q", nodes, top_n=2)
    assert [n.node.id_ for n in out] == ["b", "a"]
    assert out[0].score == pytest.approx(0.9526, abs=1e-3)
    assert out[1].score == pytest.approx(0.1192, abs=1e-3)
    assert all(0.0 <= n.score <= 1.0 for n in out)

def test_sigmoid_is_requested_explicitly(stub):
    """Pinning the activation at the call site is what replaced _force_identity_activation():
    sentence-transformers otherwise picks Sigmoid or Identity per model."""
    import torch
    connector = st_mod.SentenceTransformerRerankConnector(Settings(_env_file=None))
    connector.rerank("q", [NodeWithScore(node=TextNode(text="a", id_="a"))], top_n=1)
    assert isinstance(stub.last_activation, torch.nn.Sigmoid)

def test_top_n_truncates(stub):
    connector = st_mod.SentenceTransformerRerankConnector(Settings(_env_file=None))
    nodes = [NodeWithScore(node=TextNode(text=t, id_=t)) for t in ("a", "b")]
    assert len(connector.rerank("q", nodes, top_n=1)) == 1

def test_warmup_loads_the_model_up_front(stub):
    """app_lifespan calls this so the first query does not pay the weight-loading cost.
    Constructing the connector must not load anything by itself."""
    connector = st_mod.SentenceTransformerRerankConnector(Settings(_env_file=None))
    assert st_mod._cross_encoder.cache_info().currsize == 0
    connector.warmup()
    assert st_mod._cross_encoder.cache_info().currsize == 1
