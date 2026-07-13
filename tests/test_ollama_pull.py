"""Tests for on-demand Ollama model pulling (auto-pull when a selected model is not present)."""
import pytest
import httpx
import src.providers.ollama as ol

@pytest.fixture(autouse=True)
def _clear_cache():
    """ensure_model is lru_cached per process; reset between tests so each starts clean."""
    ol.ensure_model.cache_clear()
    yield
    ol.ensure_model.cache_clear()

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
    def json(self):
        return self._payload
    def raise_for_status(self):
        pass

def test_present_matches_bare_name_against_latest_tag(monkeypatch):
    """A bare 'nomic-embed-text' is satisfied by Ollama's stored 'nomic-embed-text:latest'."""
    monkeypatch.setattr(ol.httpx, "get", lambda *a, **k: _FakeResp({"models": [{"name": "nomic-embed-text:latest"}]}))
    assert ol._present("http://x:11434", "nomic-embed-text") is True

def test_present_requires_exact_tag_when_tag_given(monkeypatch):
    """An explicit tag must match exactly -- gemma3:270m is NOT satisfied by gemma3:1b."""
    monkeypatch.setattr(ol.httpx, "get", lambda *a, **k: _FakeResp({"models": [{"name": "gemma3:1b"}]}))
    assert ol._present("http://x:11434", "gemma3:270m") is False

def test_ensure_model_pulls_when_absent(monkeypatch):
    """Missing model -> a single POST /api/pull is issued."""
    monkeypatch.setattr(ol, "_present", lambda api, m: False)
    pulled = {}
    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json):
            pulled["url"] = url; pulled["name"] = json["name"]
            return _FakeResp({})
    monkeypatch.setattr(ol.httpx, "Client", _Client)
    ol.ensure_model("http://x:11434", "nomic-embed-text")
    assert pulled["name"] == "nomic-embed-text"
    assert pulled["url"].endswith("/api/pull")

def test_ensure_model_skips_when_present(monkeypatch):
    """Present model -> no pull attempted."""
    monkeypatch.setattr(ol, "_present", lambda api, m: True)
    def _boom(*a, **k):
        raise AssertionError("should not pull when the model is already present")
    monkeypatch.setattr(ol.httpx, "Client", _boom)
    ol.ensure_model("http://x:11434", "gemma3:270m")  # must not raise

def test_ensure_model_swallows_errors(monkeypatch):
    """A pull/connection failure is logged, not raised -- it surfaces later at call time."""
    def _boom(*a, **k):
        raise httpx.ConnectError("no ollama")
    monkeypatch.setattr(ol, "_present", _boom)
    ol.ensure_model("http://x:11434", "whatever")  # must not raise
