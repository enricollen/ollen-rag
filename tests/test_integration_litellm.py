"""Live tests for the LiteLLM connectors. Run explicitly: pytest -m integration

The watsonx parity test is the load-bearing one: it checks that routing watsonx through LiteLLM
does not regress to the raw text-generation endpoint, whose missing chat template makes instruct
models ramble past the answer (see WatsonxConnector's docstring).
"""
import httpx
import pytest
import src.providers.llm.litellm as lite_mod
import src.providers.llm.watsonx as wx_mod
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