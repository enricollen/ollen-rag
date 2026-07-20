"""First-run onboarding support: decide whether the service is configured enough to leave the
wizard, and probe a provider with candidate credentials without persisting anything."""
from functools import lru_cache
from src.settings import Settings

# Providers that require no credentials to function (local / on-box).
_KEYLESS_LLM = {"litellm-ollama"}
_KEYLESS_EMBEDDING = {"fastembed", "litellm-ollama"}

@lru_cache
def detected_compute() -> str:
    """'gpu' when the baked torch build sees a CUDA device, else 'cpu'. Read-only signal for the
    wizard: the torch flavor is fixed at image build time (TORCH_FLAVOR build arg), not switchable
    at runtime, so this only reports what was built -- it never changes it."""
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _llm_ready(settings: Settings) -> bool:
    """True when the active LLM provider has everything it needs. Keyless local providers always
    qualify; cloud providers require their credential fields to be non-empty. An empty provider
    (nothing chosen yet) is never ready."""
    provider = settings.llm_provider
    if provider in _KEYLESS_LLM:
        return True
    if provider in ("watsonx", "litellm-watsonx"):
        return bool(settings.watsonx_apikey and settings.watsonx_project_id)
    if provider == "litellm-openai":
        return bool(settings.openai_model)
    if provider == "litellm-openrouter":
        return bool(settings.openrouter_model)
    if provider.startswith("litellm"):
        return bool(settings.litellm_model and (settings.litellm_api_key or settings.litellm_api_base))
    return False

def _embedding_ready(settings: Settings) -> bool:
    """True when the active embedding provider has everything it needs. Mirrors _llm_ready with
    the embedding-specific field names (e.g. openai_embedding_model, not openai_model) -- the two
    are independent selections and one being ready says nothing about the other."""
    provider = settings.embedding_provider
    if provider in _KEYLESS_EMBEDDING:
        return True
    if provider in ("watsonx", "litellm-watsonx"):
        return bool(settings.watsonx_apikey and settings.watsonx_project_id)
    if provider == "litellm-openai":
        return bool(settings.openai_embedding_model)
    if provider == "litellm-openrouter":
        return bool(settings.openrouter_embedding_model)
    if provider.startswith("litellm"):
        return bool(
            settings.litellm_embedding_model
            and (settings.effective_litellm_embedding_api_key or settings.effective_litellm_embedding_api_base)
        )
    return False

def is_configured(settings: Settings) -> bool:
    """True when both the LLM and the embedding provider have everything they need to run a full
    ingest -> retrieve -> generate cycle. The reranker is excluded: its default (a local
    cross-encoder) is always keyless, so a misconfigured cloud reranker fails at query time with a
    clear provider error rather than blocking onboarding on a third, rarely-changed setting."""
    return _llm_ready(settings) and _embedding_ready(settings)

def needs_wizard(settings: Settings) -> bool:
    """True only for a virgin install (no LLM provider chosen yet).

    Once the operator has picked a provider — even if a later Settings edit leaves embeddings
    incomplete — they stay in the console. Re-running the full first-run wizard on every F5 after
    a partial settings save is far more annoying than a soft 'not ready' banner."""
    return not bool(settings.llm_provider)

def _probe_llm(settings: Settings) -> None:
    """Run a minimal live generation to prove the LLM credentials work. Raises on failure."""
    from src.factories.llm import create_llm  # lazy: pulls provider registrations
    llm = create_llm(settings)
    llm.complete("ping")  # tiny call; any auth/network error raises here

def _probe_embedding(settings: Settings) -> None:
    """Embed one short string to prove embedding credentials work. Raises on failure."""
    from src.factories.embeddings import create_embedding_model
    create_embedding_model(settings).get_text_embedding("ping")

def probe(target: str, settings: Settings) -> None:
    """Dispatch to the right probe. Raises on failure; returns None on success."""
    if target == "embedding":
        _probe_embedding(settings)
    else:
        _probe_llm(settings)
