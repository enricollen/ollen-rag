"""Everything about which model belongs to which provider, in one place.

Two questions live here, both keyed by provider name:

- *Which models may a caller ask for?* -- the curated yaml catalogs behind the UI dropdowns and the
  per-request validation. A provider mapped to an empty list accepts any model string, which is how
  the generic "litellm" provider reaches a new vendor without a code or config change (see
  GenericLiteLLMConnector's docstring).
- *Which model is configured right now?* -- resolved through the Settings field each provider
  declares at registration time (its model_field).

The second group is why this module exists as more than a yaml loader. That provider -> field
mapping used to be hand-copied into ingestion, retrieval, and two UI pages; the copies drifted, and
the ones written as `if provider == "watsonx" else ...` reported the wrong model instead of raising.
The factories hold the mapping (they collect it at registration) and delegate the logic here, so
there is exactly one implementation of it.
"""
from functools import lru_cache
from pathlib import Path
import yaml
from src.settings import Settings

EMBEDDING_MODELS_CONFIG_PATH = Path("config/embedding_models.yaml")
RERANKER_MODELS_CONFIG_PATH = Path("config/reranker_models.yaml")

@lru_cache
def load_model_choices(path: Path) -> dict[str, list[str]]:
    """Parse a catalog yaml into provider -> [model id]. A null/absent list normalizes to []."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {provider: list(models or []) for provider, models in raw.items()}

def validate_model(choices: dict[str, list[str]], provider: str, model: str | None) -> None:
    """Reject an unknown provider, or a model outside that provider's curated list.

    model=None means "no override requested", which is always valid. A provider with an empty
    curated list accepts any model string (see module docstring).
    """
    allowed = choices.get(provider)
    if allowed is None:
        raise ValueError(f"Unknown provider '{provider}'. Available: {sorted(choices)}")
    if allowed and model and model not in allowed:
        raise ValueError(f"Unknown model '{model}' for provider '{provider}'. Available: {allowed}")

def model_field(model_fields: dict[str, str], provider: str) -> str:
    """Name of the Settings field holding *provider*'s model id."""
    if provider not in model_fields:
        raise ValueError(f"Unknown provider '{provider}'. Available providers: {sorted(model_fields)}")
    return model_fields[provider]

def resolve_model(settings: Settings, provider_field: str, model_fields: dict[str, str], provider: str | None = None) -> str:
    """The model id configured for *provider* (default: whichever provider settings selects).

    *provider_field* names the Settings field holding the active provider, e.g. "embedding_provider".
    """
    provider = provider or getattr(settings, provider_field)
    return getattr(settings, model_field(model_fields, provider))

def with_model(
    settings: Settings, provider_field: str, model_fields: dict[str, str], provider: str, model: str | None = None,
) -> Settings:
    """A Settings copy pinned to *provider*, and to *model* when one is given.

    The write path: a per-request override, or an index's recorded build config, has to land in the
    field that provider actually reads -- not in some generic slot, which does not exist.
    """
    overrides: dict[str, str] = {provider_field: provider}
    if model:
        overrides[model_field(model_fields, provider)] = model
    return settings.model_copy(update=overrides)

def default_models(settings: Settings, model_fields: dict[str, str]) -> dict[str, str]:
    """provider -> its currently configured model, for the UI's /api/v1/config payload."""
    return {provider: getattr(settings, field) for provider, field in model_fields.items()}
