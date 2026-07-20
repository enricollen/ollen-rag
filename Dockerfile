# Dockerfile — self-contained image for normal (non-OpenShift) hosts.

# ---- web-builder: React console (frontend/) -> static assets. Discarded after the multi-stage
# copy below, so node/npm/node_modules never end up in the final image. ----
FROM node:22-slim AS web-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim

# Cache locations under the app home so baked model weights and HF downloads have a stable,
# writable path in both the build and the running container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# uv binary only — no need for a full uv base image on the runtime stage.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

# LibreOffice + ImageMagick are required by liteparse for Office/image formats. curl for healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-nogui imagemagick curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./

# Torch flavor: "cpu" (default) or "gpu". sentence-transformers pulls torch, whose default PyPI wheel
# bundles ~5-6GB of NVIDIA CUDA libraries. On a CPU-only host that is pure bloat, so by default we
# reinstall the CPU wheel after sync. Build with `--build-arg TORCH_FLAVOR=gpu` (and run with
# `--gpus all`) to keep the CUDA build instead.
ARG TORCH_FLAVOR=cpu
RUN uv sync --frozen --no-dev --no-install-project \
    && if [ "$TORCH_FLAVOR" = "cpu" ]; then \
         uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cpu; \
       fi

# Flatten backend/ into /app so uvicorn app:app and relative paths (config/, .env) stay unchanged.
COPY backend/ .
# Overwrite whatever (if anything) a local frontend/dist snuck into the context with the fresh
# build -- the console served at /ui/ always comes from this build, never a stale local one.
COPY --from=web-builder /frontend/dist ./frontend/dist

# Bake the two default local models so first boot works offline with no download stall.
# Reranker cross-encoder -> HF cache; fastembed model -> its cache dir (matches settings default).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')" \
    && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='./.cache/fastembed')"

# Plain non-root user (no OpenShift arbitrary-UID gymnastics); owns the app + caches it writes to.
# Pre-create the volume mount points (config store + chroma data) so a fresh named volume mounted
# here inherits appuser ownership -- otherwise Docker creates them root-owned and the non-root
# process cannot write its .env / index files.
RUN mkdir -p /app/config-store /app/data/chroma \
    && useradd --create-home --home-dir /home/appuser appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
