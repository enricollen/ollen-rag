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
    assert r.json() == {"restarting": True, "restart_mode": "reload"}
    assert env.read_text() == "OLLEN_RAG_CHUNK_SIZE=256"
    assert touched.get("hit") is True