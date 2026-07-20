"""Service entrypoint: FastAPI app with the FastMCP server mounted at /mcp."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastmcp.utilities.lifespan import combine_lifespans
from src.api.routes import router
from src.exceptions import OllenRagError
from src.factories.vector_store import create_backend
from src.logger import OllenLogger
from src.mcp_server import mcp
from src.factories.reranker import create_reranker
from src.settings import get_settings

logger = OllenLogger("app")

def _ui_dist() -> str:
    """Resolve the React console build dir for both layouts:
    - docker: backend is flattened into /app, so frontend/dist sits next to app.py
    - local:  app.py lives in backend/, so frontend/dist is one level up at the repo root
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / "frontend" / "dist", here.parent / "frontend" / "dist"):
        if candidate.is_dir():
            return str(candidate)
    return str(here / "frontend" / "dist")


class _QuietPollFilter(logging.Filter):
    """Drop uvicorn access-log lines for high-frequency liveness/status polls. The UI hits /health
    every 15s and the onboarding status endpoint on every page load, so their access lines bury the
    requests that matter (ingestion, retrieval, eval). Real endpoints stay logged."""
    _QUIET = ("/health", "/api/v1/onboarding/status")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self._QUIET)

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Warm up shared resources at startup so first requests aren't slow."""
    started = perf_counter()
    logger.debug("warming up vector store backend (%s)", get_settings().vector_store)
    try:
        create_backend(get_settings()).warmup()  # e.g. OpenSearch: ensure the hybrid search pipeline
        logger.info("vector store backend ready (%.2fs)", perf_counter() - started)
    except Exception as exc:
        logger.warning("could not warm up vector store backend at startup: %s", exc)
    started = perf_counter()
    logger.debug("pre-loading reranker model")
    try:
        # Building the connector is cheap; warmup() is what pulls a local model's weights off
        # disk. Both are cached for the process lifetime.
        create_reranker().connector.warmup()
        logger.info("reranker model loaded (%.2fs)", perf_counter() - started)
    except Exception as exc:
        logger.warning("could not pre-load reranker at startup: %s", exc)
    logger.info("startup complete — ready to serve")
    from src.rag.onboarding import is_configured
    if is_configured(get_settings()):
        logger.info("console: http://localhost:8000/ui/")
    else:
        logger.info("setup required — open http://localhost:8000/ui/ to pick a provider, or set "
                     "OLLEN_RAG_LLM_PROVIDER / OLLEN_RAG_EMBEDDING_PROVIDER and restart")
    yield

def create_app() -> FastAPI:
    """Assemble the FastAPI application: REST routes, error handlers, mounted MCP server."""
    OllenLogger.setup(get_settings())
    # Silence the repetitive /health + status polls in uvicorn's access log (real requests stay).
    logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())
    s = get_settings()
    # Show the logo first, then the resolved config line below it, as the boot header.
    OllenLogger.banner(f"v0.1.0 · log level {s.log_level.upper()}")
    # Resolve each component to the model its *active provider* actually uses (same source the UI
    # banner reads), so the startup line reflects the live config rather than a fixed field.
    import src.providers  # noqa: F401  populate registries before the factory model lookups
    from src.config.summary import component_summary
    a = component_summary(s)
    logger.info(
        "components: llm=%s/%s embedding=%s/%s reranker=%s/%s vector_store=%s "
        "chunking=%s(size=%d,overlap=%d) rerank_top_n=%d retrieval_top_k=%d",
        a["llm"]["provider"], a["llm"]["model"],
        a["embedding"]["provider"], a["embedding"]["model"],
        a["reranker"]["provider"], a["reranker"]["model"], a["vector_store"],
        a["chunking"]["strategy"], a["chunking"]["chunk_size"], a["chunking"]["chunk_overlap"],
        a["rerank_top_n"], a["retrieval_top_k"],
    )
    mcp_app = mcp.http_app(path="/")
    app = FastAPI(
        title="ollen-rag-service",
        version="0.1.0",
        lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan),
    )
    app.include_router(router)
    # First-run setup wizard endpoints (status + provider credential test).
    from src.api.onboarding import router as onboarding_router
    app.include_router(onboarding_router)
    app.mount("/mcp", mcp_app)
    # React console (frontend/), built by the Dockerfile's web-builder stage / `npm run build` in
    # Plain Python mode -- served at /ui/. Hash-routed SPA, so html=True's index.html fallback
    # only ever needs to answer GET /ui/ itself.
    app.mount("/ui", StaticFiles(directory=_ui_dist(), html=True), name="ui")

    @app.middleware("http")
    async def ui_shell_no_cache(request: Request, call_next):
        # index.html must not be cached -- it points at hashed asset filenames; a stale shell keeps
        # serving an old bundle even after a rebuild.
        response = await call_next(request)
        path = request.url.path.rstrip("/")
        if path in ("/ui",) or request.url.path.endswith("/ui/") or request.url.path.endswith("/index.html"):
            if "/ui" in request.url.path:
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
        return response

    @app.exception_handler(OllenRagError)
    async def ollen_rag_error_handler(request: Request, exc: OllenRagError) -> JSONResponse:
        # Map domain errors to their HTTP status with a stable machine-readable code
        return JSONResponse(status_code=exc.status_code, content={"error_code": exc.error_code, "detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        # Invalid strategy/provider names surface as 400s
        return JSONResponse(status_code=400, content={"error_code": "INVALID_REQUEST", "detail": str(exc)})

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # reload=True requires the app passed as an import string (uvicorn re-imports it in the
    # watched worker subprocess); the `app` object built above is only used by other ASGI
    # servers/import paths (e.g. `uvicorn app:app` without --reload, gunicorn workers).
    # This is what makes the Settings UI's "Save & restart" (config/restart.py) actually take
    # effect under a plain `python app.py`: it touches this file's mtime, and reload's file
    # watcher is what turns that touch into a worker respawn that re-reads .env.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
