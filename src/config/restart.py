"""Mode-aware config-apply restart. The service must re-read .env after the wizard writes it,
in three runtimes: `reload` (uvicorn --reload watches files), `exit` (Docker restart policy
reboots an exited process), and `manual` (plain process, no supervisor -> user restarts)."""
import os
from pathlib import Path

def _reloader_active() -> bool:
    """True when uvicorn's reloader is managing this process. The reloader launches the app in a
    child and sets this marker in the child's environment; its presence is the reliable signal."""
    return os.environ.get("UVICORN_RELOAD_PROCESS") == "true" or "--reload" in os.environ.get("_", "")

def resolve_restart_mode(configured: str) -> str:
    """Pick the apply strategy. An explicit OLLEN_RAG_RESTART_MODE wins; otherwise infer:
    reloader present -> 'reload' (keep today's behavior), else 'manual' (never kill an
    unrestartable plain process)."""
    if configured in {"reload", "exit", "manual"}:
        return configured
    return "reload" if _reloader_active() else "manual"

def _touch_app() -> None:
    """Bump app.py's mtime so `uvicorn --reload` (which watches *.py, not .env) restarts the
    worker, which then re-reads the freshly written .env from a new get_settings()."""
    Path("app.py").touch()

def apply_restart(mode: str) -> None:
    """Execute the chosen strategy. `exit` hard-exits so Docker's restart policy reboots the
    container with the new .env; `reload` touches a watched file; `manual` does nothing."""
    if mode == "reload":
        _touch_app()
    elif mode == "exit":
        os._exit(0)
    # manual: intentionally nothing -- the caller tells the user to restart.
