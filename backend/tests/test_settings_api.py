"""Tests for the editable settings API: full read, validated write, .env merge, restart trigger."""
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from app import create_app
    with TestClient(create_app()) as c:
        yield c

def test_get_settings_returns_all_fields_incl_secret(client):
    body = client.get("/api/v1/settings").json()
    assert isinstance(body["chunk_size"], int)
    assert "watsonx_apikey" in body  # secrets exposed by design

def test_post_unknown_field_is_400(client):
    r = client.post("/api/v1/settings", json={"not_a_field": "x"})
    assert r.status_code == 400

def test_post_bad_type_is_400(client):
    r = client.post("/api/v1/settings", json={"chunk_size": "not-an-int"})
    assert r.status_code == 400

def test_post_valid_writes_env_and_triggers_restart(client, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OLLEN_RAG_CHUNK_SIZE=512")
    monkeypatch.setattr("src.api.routes.ENV_PATH", env)
    # Force the reload path (as under `uvicorn --reload`) so apply touches app.py deterministically.
    monkeypatch.setattr("src.config.restart._reloader_active", lambda: True)
    touched = {}
    monkeypatch.setattr("src.config.restart._touch_app", lambda: touched.setdefault("hit", True))
    r = client.post("/api/v1/settings", json={"chunk_size": 256})
    assert r.status_code == 200
    assert r.json() == {"restarting": True, "restart_mode": "reload", "applied_live": True}
    assert env.read_text() == "OLLEN_RAG_CHUNK_SIZE=256"
    assert touched.get("hit") is True

def test_post_valid_applies_live_without_any_restart(client, tmp_path, monkeypatch):
    """No supervisor, no reloader (the plain `docker run` / bare-process case): the change must
    still take effect for the very next request, with no process restart of any kind.

    Chdir (not just patching ENV_PATH) so pydantic-settings' own `env_file=".env"` read and the
    route's merge_into_env write target the same file -- otherwise this would only prove the
    write happened, not that get_settings() actually picks it up."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("OLLEN_RAG_RESTART_MODE", "manual")
    monkeypatch.setattr("src.config.restart._reloader_active", lambda: False)
    from src.settings import get_settings
    get_settings.cache_clear()
    r = client.post("/api/v1/settings", json={"chunk_size": 777})
    assert r.status_code == 200
    assert r.json() == {"restarting": False, "restart_mode": "manual", "applied_live": True}
    assert get_settings().chunk_size == 777

def test_post_valid_overrides_a_shadowing_process_env_var(client, tmp_path, monkeypatch):
    """A field baked in as a literal process env var (e.g. docker-compose's `environment:` block)
    would otherwise always win over .env -- saving that same field from the UI must pop it from
    os.environ so the freshly written .env value actually takes effect immediately."""
    import os
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("OLLEN_RAG_VECTOR_STORE", "chroma")
    from src.settings import get_settings
    get_settings.cache_clear()
    assert get_settings().vector_store == "chroma"
    r = client.post("/api/v1/settings", json={"vector_store": "opensearch"})
    assert r.status_code == 200
    assert get_settings().vector_store == "opensearch"
    assert os.environ.get("OLLEN_RAG_VECTOR_STORE") is None