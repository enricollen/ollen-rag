"""First-run onboarding support: decide whether the service is configured enough to leave the
wizard, and probe a provider with candidate credentials without persisting anything."""
from functools import lru_cache
from src.settings import Settings

# Providers that require no credentials to function (local / on-box).
_KEYLESS_LLM = {"litellm-ollama"}

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

def is_configured(settings: Settings) -> bool:
    """True when the active LLM provider has everything it needs. Keyless local providers always
    qualify; cloud providers require their credential fields to be non-empty. Embeddings/rerank
    default to keyless-local, so the LLM is the gating choice for leaving the wizard."""
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
