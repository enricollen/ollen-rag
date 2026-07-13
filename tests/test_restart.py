"""Tests for the mode-aware config-apply restart selector and executor."""
from src.config import restart

def test_resolve_explicit_modes():
    """An explicit OLLEN_RAG_RESTART_MODE is honored verbatim."""
    assert restart.resolve_restart_mode("exit") == "exit"
    assert restart.resolve_restart_mode("reload") == "reload"
    assert restart.resolve_restart_mode("manual") == "manual"

def test_resolve_unset_defaults_to_manual_without_reloader(monkeypatch):
    """Plain `uvicorn app:app` (no --reload, no supervisor): don't kill an unrestartable process."""
    monkeypatch.setattr(restart, "_reloader_active", lambda: False)
    assert restart.resolve_restart_mode("") == "manual"

def test_resolve_unset_defaults_to_reload_under_reloader(monkeypatch):
    """`uvicorn --reload` sets a watched-process env marker; keep today's touch behavior."""
    monkeypatch.setattr(restart, "_reloader_active", lambda: True)
    assert restart.resolve_restart_mode("") == "reload"

def test_apply_manual_is_noop(monkeypatch):
    """manual must never touch files or exit the process."""
    called = {"touched": False, "exited": False}
    monkeypatch.setattr(restart, "_touch_app", lambda: called.__setitem__("touched", True))
    monkeypatch.setattr(restart.os, "_exit", lambda code: called.__setitem__("exited", True))
    restart.apply_restart("manual")
    assert called == {"touched": False, "exited": False}

def test_apply_reload_touches_app(monkeypatch):
    """reload mode bumps app.py's mtime so the uvicorn reloader reboots the worker."""
    touched = {"v": False}
    monkeypatch.setattr(restart, "_touch_app", lambda: touched.__setitem__("v", True))
    restart.apply_restart("reload")
    assert touched["v"] is True
