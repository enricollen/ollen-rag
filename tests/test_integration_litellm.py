"""Live tests for the LiteLLM connectors. Run explicitly: pytest -m integration

The watsonx parity tests are the load-bearing ones. For generation, parity means not regressing to
the raw text-generation endpoint, whose missing chat template makes instruct models ramble past the
answer (see WatsonxConnector's docstring). For embeddings, it means the LiteLLM path lands in the
same vector space as the native SDK, so one index can be queried through either provider.

The rerank test is also what proves, against the live service, the two SDK claims the design rests
on: that litellm.rerank forwards project_id as a call kwarg (no WATSONX_* env vars needed), and
that response.results entries are subscriptable dicts.
"""
import httpx
import pytest
from llama_index.core.schema import NodeWithScore, TextNode
import src.providers.llm.litellm as lite_mod
import src.providers.llm.watsonx as wx_mod
from src.factories.embeddings import create_embedding_model
from src.factories.reranker import create_reranker
from src.settings import get_settings

# Mark every test in this module as integration-only so pytest.ini's
# `addopts = -m "not integration"` excludes it from the default run.
pytestmark = pytest.mark.integration

PROMPT = "Rispondi in una sola frase: qual e la capitale d'Italia?"

def _has_role_labels(text: str) -> bool:
    """Detect the raw text-generation-endpoint pathology: the model fabricating its own chat turns."""
    lowered = text.lower()
    return "user:" in lowered or "assistant:" in lowered

def _ollama_up(api_base: str) -> bool:
    """Ollama exposes an unauthenticated root endpoint; use it as a cheap reachability probe."""
    try:
        return httpx.get(api_base, timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False

@pytest.fixture(scope="module")
def settings():
    """Real settings from .env -- these tests need real credentials by definition."""
    get_settings.cache_clear()
    return get_settings()

def test_native_watsonx_answers_cleanly(settings):
    """Baseline: establishes what correct watsonx output looks like on this account and model."""
    if not settings.watsonx_apikey or not settings.watsonx_project_id:
        pytest.skip("watsonx credentials not configured in .env")
    answer = wx_mod.WatsonxConnector(settings=settings).complete(PROMPT)
    assert answer.strip()
    assert not _has_role_labels(answer)

def test_litellm_watsonx_matches_native_behaviour(settings):
    """Parity: LiteLLM's watsonx path must not regress to the raw text-generation endpoint."""
    if not settings.watsonx_apikey or not settings.watsonx_project_id:
        pytest.skip("watsonx credentials not configured in .env")
    answer = lite_mod.LiteLLMWatsonxConnector(settings=settings).complete(PROMPT)

    assert answer.strip(), "LiteLLM returned an empty completion"
    assert not _has_role_labels(answer), (
        f"LiteLLM's watsonx path produced fabricated chat turns, which means it hit the raw "
        f"text-generation endpoint rather than /ml/v1/text/chat. Answer was: {answer!r}"
    )
    # A one-sentence answer that ran to max_new_tokens was truncated, not completed.
    assert len(answer.split()) < settings.watsonx_max_new_tokens

def test_litellm_ollama_completes(settings):
    """Round-trips a real completion through a local Ollama, if one is running."""
    if not _ollama_up(settings.ollama_api_base):
        pytest.skip(f"Ollama not reachable at {settings.ollama_api_base}; run: ollama serve")
    answer = lite_mod.LiteLLMOllamaConnector(settings=settings).complete(PROMPT)
    assert answer.strip()

# --- embeddings ---

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, without pulling numpy into the test."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm

def test_litellm_watsonx_embedding_matches_native_vector_space(settings):
    """Parity: an index built by either provider must be queryable by the other.

    Identical model id, identical endpoint, so the vectors must coincide -- not merely have the
    same length. A mismatch here would silently mix vector spaces at query time.
    """
    if not settings.watsonx_apikey or not settings.watsonx_project_id:
        pytest.skip("watsonx credentials not configured in .env")
    text = "La capitale della Francia e Parigi."
    native = create_embedding_model(settings.model_copy(update={"embedding_provider": "watsonx"}))
    lite = create_embedding_model(settings.model_copy(update={"embedding_provider": "litellm-watsonx"}))
    native_vec = native.get_text_embedding(text)
    lite_vec = lite.get_text_embedding(text)
    assert len(lite_vec) == len(native_vec)
    assert _cosine(native_vec, lite_vec) > 0.999

def test_litellm_ollama_embedding_returns_a_vector(settings):
    """Round-trips a real embedding through a local Ollama, if one is running."""
    if not _ollama_up(settings.ollama_api_base):
        pytest.skip(f"Ollama not reachable at {settings.ollama_api_base}; run: ollama serve")
    model = create_embedding_model(settings.model_copy(update={"embedding_provider": "litellm-ollama"}))
    vector = model.get_text_embedding("prova di embedding")
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)

# --- rerank ---

def _rerank_nodes() -> list[NodeWithScore]:
    """Two passages where the right answer is unambiguous, so ordering is a real assertion."""
    return [
        NodeWithScore(node=TextNode(text="Roma e la capitale d'Italia.", id_="roma"), score=0.5),
        NodeWithScore(node=TextNode(text="Parigi e la capitale della Francia.", id_="parigi"), score=0.5),
    ]

def test_litellm_watsonx_rerank_returns_probabilities(settings):
    """The 0-1 contract, against the real endpoint, best first.

    Also the live proof that project_id survives as a call kwarg and that response.results
    entries are subscriptable -- both verified against the SDK source, neither exercised by the
    monkeypatched unit tests.
    """
    if not settings.watsonx_apikey or not settings.watsonx_project_id:
        pytest.skip("watsonx credentials not configured in .env")
    reranker = create_reranker(top_n=2, provider="litellm-watsonx", settings=settings)
    out = reranker.postprocess_nodes(_rerank_nodes(), query_str="Qual e la capitale della Francia?")
    assert [n.node.id_ for n in out] == ["parigi", "roma"]
    assert all(0.0 <= n.score <= 1.0 for n in out)
    assert out[0].score > out[1].score

def test_sentence_transformers_rerank_returns_probabilities(settings):
    """The same contract for the local cross-encoder, so the two providers are interchangeable.
    Downloads the model on first run."""
    reranker = create_reranker(top_n=2, provider="sentence-transformers", settings=settings)
    out = reranker.postprocess_nodes(_rerank_nodes(), query_str="Qual e la capitale della Francia?")
    assert [n.node.id_ for n in out] == ["parigi", "roma"]
    assert all(0.0 <= n.score <= 1.0 for n in out)
    assert out[0].score > out[1].score