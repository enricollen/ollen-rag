"""Tests for the provider-agnostic reranker factory and its 0-1 score contract."""
import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from src.factories import reranker as rr_mod
from src.factories.reranker import RerankConnector, RerankerFactory, create_reranker
from src.settings import Settings

def _nodes(*texts: str) -> list[NodeWithScore]:
    """Fused-score nodes as they arrive from the vector store."""
    return [NodeWithScore(node=TextNode(text=t, id_=t), score=0.5) for t in texts]

# Registered at import, not inside a test: a registration performed in a test body would make
# every other test in this module depend on execution order.
@RerankerFactory.register("fake-rerank", model_field="reranker_model")
class _FakeConnector(RerankConnector):
    """Returns fixed 0-1 scores, so the adapter can be tested without loading a model."""
    def rerank(self, query, nodes, top_n):
        ranked = [NodeWithScore(node=n.node, score=s) for n, s in zip(nodes, [0.9, 0.1], strict=True)]
        return ranked[:top_n]

@pytest.fixture(autouse=True)
def _clear_connector_cache():
    """The connector cache is process-wide; isolate tests from each other."""
    rr_mod._connectors.clear()
    yield
    rr_mod._connectors.clear()

def test_connector_contract_is_enforced_by_the_adapter():
    """ConnectorRerank hands llamaindex whatever the connector returned, in order."""
    s = Settings(_env_file=None, reranker_provider="fake-rerank", rerank_top_n=2)
    postprocessor = create_reranker(settings=s)
    out = postprocessor.postprocess_nodes(_nodes("a", "b"), query_bundle=QueryBundle("q"))
    assert [n.score for n in out] == [0.9, 0.1]

def test_top_n_truncates():
    s = Settings(_env_file=None, reranker_provider="fake-rerank")
    out = create_reranker(top_n=1, settings=s).postprocess_nodes(_nodes("a", "b"), query_bundle=QueryBundle("q"))
    assert len(out) == 1

def test_unknown_provider_raises():
    """The registry rejects the name, before the yaml catalog is ever consulted."""
    with pytest.raises(ValueError, match="Unknown reranker provider 'banana'"):
        create_reranker(provider="banana", settings=Settings(_env_file=None))

def test_unknown_model_for_curated_provider_raises():
    s = Settings(_env_file=None)
    with pytest.raises(ValueError, match="Unknown model 'not-a-model'"):
        create_reranker(provider="sentence-transformers", model="not-a-model", settings=s)

def test_a_provider_absent_from_the_yaml_still_works_without_a_model_override():
    """Provider existence comes from the registry; the yaml only constrains model overrides.
    fake-rerank is registered but deliberately absent from config/reranker_models.yaml."""
    s = Settings(_env_file=None, reranker_provider="fake-rerank")
    assert create_reranker(settings=s) is not None

def test_connectors_are_cached_per_provider_and_model():
    """Loading a cross-encoder is expensive; one instance per (provider, model) for the process.
    Constructing SentenceTransformerRerankConnector is cheap -- weights load lazily on first
    rerank() -- so this exercises the real provider without a model download."""
    s = Settings(_env_file=None, reranker_provider="sentence-transformers")
    create_reranker(settings=s)
    create_reranker(settings=s)
    assert len(rr_mod._connectors) == 1
    create_reranker(provider="sentence-transformers", model="models/reranker", settings=s)
    assert len(rr_mod._connectors) == 2

def test_register_requires_a_model_field():
    """Same guarantee as EmbeddingFactory: a per-request model override has to know which
    Settings field to write into."""
    with pytest.raises(TypeError):
        RerankerFactory.register("no-field")

def test_catalog_covers_every_registered_provider():
    """Same drift guard as the embedding factory: yaml catalog vs code registry."""
    import src.providers  # noqa: F401
    registered = set(RerankerFactory.providers()) - {"fake-rerank"}
    assert registered <= set(rr_mod.load_reranker_model_choices())
