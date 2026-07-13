"""Tests for the first-run onboarding endpoints (status + provider probe) and its config gate."""
from fastapi.testclient import TestClient
from app import app
from src.settings import Settings
from src.rag.onboarding import is_configured

client = TestClient(app)

def test_is_configured_false_on_empty_watsonx():
    """Default provider watsonx with no apikey is not yet usable."""
    s = Settings(_env_file=None, llm_provider="watsonx", watsonx_apikey="", watsonx_project_id="")
    assert is_configured(s) is False

def test_is_configured_true_for_ollama():
    """Local Ollama needs no credentials, so a fresh install with it selected is 'configured'."""
    s = Settings(_env_file=None, llm_provider="litellm-ollama")
    assert is_configured(s) is True

def test_status_endpoint_shape():
    resp = client.get("/api/v1/onboarding/status")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"configured", "llm_provider", "vector_store", "compute"}
    assert body["compute"] in {"cpu", "gpu"}

def test_test_endpoint_reports_failure_cleanly(monkeypatch):
    """A provider probe that raises must become {ok: false, detail: <msg>}, never a 500."""
    import src.rag.onboarding as ob
    monkeypatch.setattr(ob, "_probe_llm", lambda s: (_ for _ in ()).throw(RuntimeError("bad key")))
    resp = client.post("/api/v1/onboarding/test", json={"target": "llm", "changes": {"llm_provider": "watsonx"}})
    assert resp.status_code == 200
    assert resp.json() == {"ok": False, "detail": "bad key"}

def test_test_endpoint_success(monkeypatch):
    import src.rag.onboarding as ob
    monkeypatch.setattr(ob, "_probe_llm", lambda s: None)  # no exception == success
    resp = client.post("/api/v1/onboarding/test", json={"target": "llm", "changes": {"llm_provider": "litellm-ollama"}})
    assert resp.json()["ok"] is True
