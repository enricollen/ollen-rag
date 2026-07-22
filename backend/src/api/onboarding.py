"""REST endpoints backing the first-run setup wizard: report configuration status and test a
provider's credentials before the user commits them to .env."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.settings import get_settings, Settings
from src.rag import onboarding

router = APIRouter(prefix="/api/v1/onboarding")

class TestRequest(BaseModel):
    """A candidate provider selection + credentials to validate without saving."""
    target: str = "llm"           # "llm" | "embedding" | "reranker"
    changes: dict = {}            # settings fields to overlay on the current config

@router.get("/status")
def status() -> dict:
    """Setup state for the console: `needs_wizard` gates the first-run UI (virgin install only);
    `configured` is the stricter ready check (/ready, soft banner). Also returns the current
    selections and the detected compute (cpu/gpu) baked into the image."""
    s = get_settings()
    return {
        "configured": onboarding.is_configured(s),
        "needs_wizard": onboarding.needs_wizard(s),
        "llm_provider": s.llm_provider,
        "embedding_provider": s.embedding_provider,
        "vector_store": s.vector_store,
        "compute": onboarding.detected_compute(),
    }

@router.post("/test")
def test(req: TestRequest) -> dict:
    """Overlay candidate changes onto current settings, build the provider, run a tiny live call.
    Returns {ok, detail}; a probe failure is a normal {ok: false} result, not an HTTP error."""
    unknown = set(req.changes) - set(Settings.model_fields)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown settings: {sorted(unknown)}")
    merged = {**get_settings().model_dump(), **req.changes}
    try:
        candidate = Settings(_env_file=None, **merged)
        onboarding.probe(req.target, candidate)
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}
    return {"ok": True, "detail": "ok"}
