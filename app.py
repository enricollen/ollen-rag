"""Service entrypoint: FastAPI app with the FastMCP server mounted at /mcp."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastmcp.utilities.lifespan import combine_lifespans
from src.api.routes import router
from src.exceptions import OllenRagError
from src.factories.vector_store import create_backend
from src.logging_config import OllenLogger
from src.mcp_server import mcp
from src.rag.retrieval import get_reranker
from src.settings import get_settings

logger = OllenLogger("app")

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Warm up shared resources at startup so first requests aren't slow."""
    try:
        create_backend(get_settings()).warmup()  # e.g. OpenSearch: ensure the hybrid search pipeline
        logger.info("Vector store backend warmed up")
    except Exception as exc:
        logger.warning("Could not warm up vector store backend at startup: %s", exc)
    try:
        get_reranker()  # loads cross-encoder weights once; cached for the process lifetime
        logger.info("Reranker model loaded")
    except Exception as exc:
        logger.warning("Could not pre-load reranker at startup: %s", exc)
    yield

def create_app() -> FastAPI:
    """Assemble the FastAPI application: REST routes, error handlers, mounted MCP server."""
    OllenLogger.setup(get_settings())
    s = get_settings()
    logger.info(
        "components: llm=%s/%s embedding=%s/%s reranker=%s "
        "chunking=%s(size=%d,overlap=%d) rerank_top_n=%d retrieval_top_k=%d",
        s.llm_provider, s.watsonx_llm_model_id,
        s.embedding_provider, s.watsonx_embedding_model_id if s.embedding_provider == "watsonx" else s.fastembed_model_name,
        s.reranker_model,
        s.default_chunking_strategy, s.chunk_size, s.chunk_overlap,
        s.rerank_top_n, s.retrieval_top_k,
    )
    mcp_app = mcp.http_app(path="/")
    app = FastAPI(
        title="ollen-rag-service",
        version="0.1.0",
        lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan),
    )
    app.include_router(router)
    app.mount("/mcp", mcp_app)
    # Manual e2e test UI (static, no build step) served at /ui/
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
