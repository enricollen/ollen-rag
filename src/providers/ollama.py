"""On-demand Ollama model pulling. The bundled Ollama pre-pulls only the default chat model; when a
user selects any other Ollama model (e.g. an embedding model like nomic-embed-text) we pull it the
first time it is needed, rather than failing with 'model not found, try pulling it first'."""
from functools import lru_cache
import httpx
from src.logger import OllenLogger

logger = OllenLogger("ollama")

def _present(api_base: str, model: str) -> bool:
    """True if `model` is already in Ollama's local list. A bare name (no ':tag') is satisfied by any
    tag of it (Ollama stores 'name:latest'); an explicit tag must match exactly, so gemma3:270m is
    not considered present just because gemma3:1b is."""
    resp = httpx.get(f"{api_base.rstrip('/')}/api/tags", timeout=10)
    resp.raise_for_status()
    names = [m.get("name", "") for m in resp.json().get("models", [])]
    if ":" in model:
        return model in names
    return any(n == model or n.startswith(model + ":") for n in names)

@lru_cache(maxsize=None)
def ensure_model(api_base: str, model: str) -> None:
    """Pull `model` into Ollama if absent. Cached so the check/pull happens at most once per process
    per (api_base, model). Best-effort: any failure (Ollama down, pull error) is logged and left to
    surface at actual call time, so this never blocks construction of a provider."""
    try:
        if _present(api_base, model):
            return
        logger.info("Ollama model %s not found at %s; pulling (first use)...", model, api_base)
        # /api/pull streams progress by default; stream=False blocks until the pull completes.
        # Generous timeout: a cold pull of a multi-hundred-MB model can take a while.
        with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
            r = client.post(f"{api_base.rstrip('/')}/api/pull", json={"name": model, "stream": False})
            r.raise_for_status()
        logger.info("Ollama model %s pulled", model)
    except Exception as exc:
        logger.warning("Could not ensure Ollama model %s at %s: %s", model, api_base, exc)
