"""Tests for the OpenSearch reachability check (opensearch_reachable in the OpenSearch provider
and the /api/v1/infra/opensearch/status route)."""
import httpx
from fastapi.testclient import TestClient
from app import app
from src.providers.vector_stores import opensearch as os_mod
from src.settings import Settings

client = TestClient(app)

def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)

def test_opensearch_reachable_true_on_response(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200))
    assert os_mod.opensearch_reachable(_settings()) is True

def test_opensearch_reachable_false_on_connect_error(monkeypatch):
    def raise_connect_error(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", raise_connect_error)
    assert os_mod.opensearch_reachable(_settings()) is False

def test_status_route_reachable(monkeypatch):
    monkeypatch.setattr("src.api.routes.opensearch_reachable", lambda s, **k: True)
    resp = client.get("/api/v1/infra/opensearch/status")
    assert resp.status_code == 200
    assert resp.json() == {"reachable": True}

def test_status_route_unreachable(monkeypatch):
    monkeypatch.setattr("src.api.routes.opensearch_reachable", lambda s, **k: False)
    resp = client.get("/api/v1/infra/opensearch/status")
    assert resp.status_code == 200
    assert resp.json() == {"reachable": False}
