# Dockerfile — self-contained image for normal (non-OpenShift) hosts.
#
# Stages:
#   web-builder    — React console -> static assets (node discarded)
#   python-builder — venv + baked models (no LibreOffice; uv discarded after copy)
#   runtime        — slim app image with system deps only

# ---- web-builder -------------------------------------------------------------
FROM node:22-slim AS web-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- python-builder: deps + model bake (no LibreOffice — keeps this stage lean) ----
FROM python:3.13-slim AS python-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

# curl only for tessdata bake; LibreOffice lives in the runtime stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --home-dir /home/appuser appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app

WORKDIR /app
USER appuser

COPY --chown=appuser:appuser backend/pyproject.toml backend/uv.lock ./

# Torch flavor: "cpu" (default) or "gpu".
# CPU: skip lock's CUDA torch *and* every nvidia-* wheel (they are separate lock entries —
# `--no-install-package torch` alone still downloads ~5-6GB of nvidia-* deps), then install
# the cpu wheel. GPU: install the lock as-is.
ARG TORCH_FLAVOR=cpu
RUN set -eux; \
    echo "TORCH_FLAVOR=${TORCH_FLAVOR}"; \
    if [ "$TORCH_FLAVOR" = "cpu" ]; then \
      TORCH_VER="$(awk '/^name = "torch"$/{p=1;next} p&&/^version =/{gsub(/"/,"",$3); print $3; exit}' uv.lock)"; \
      # build: --no-install-package nvidia-cublas --no-install-package nvidia-cudnn-cu13 ...
      SKIP_NVIDIA="$(awk '/^name = "nvidia-/{gsub(/"/,"",$3); print $3}' uv.lock | sort -u | sed 's/^/--no-install-package /' | tr '\n' ' ')"; \
      uv sync --frozen --no-dev --no-install-project --no-install-package torch ${SKIP_NVIDIA}; \
      uv pip install --torch-backend=cpu "torch==${TORCH_VER}"; \
      uv pip freeze | awk -F= '/^nvidia-/ {print $1}' | xargs -r uv pip uninstall -y; \
    else \
      uv sync --frozen --no-dev --no-install-project; \
    fi; \
    rm -rf /app/.cache/uv

# Bake the two default local models so first boot works offline with no download stall.
# Reranker cross-encoder -> HF cache; fastembed model -> its cache dir (matches settings default).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')" \
    && python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='./.cache/fastembed')" \
    && rm -rf /app/.cache/uv /tmp/tmp* /tmp/huggingface* 2>/dev/null || true

# liteparse/tesseract-rs downloads eng.traineddata from github on first OCR page. bake it in so
# scanned pdfs don't fail when the container has flaky egress to github/raw.githubusercontent.com.
RUN mkdir -p /app/.tesseract-rs/tessdata \
    && curl -fsSL -o /app/.tesseract-rs/tessdata/eng.traineddata \
         https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata

# ---- runtime -----------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

# LibreOffice + ImageMagick are required by liteparse for Office/image formats. curl for healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-nogui imagemagick curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user + volume mount points *before* COPY so we never need `chown -R /app`
# (that would rewrite venv + models into a new layer and roughly double the image).
RUN useradd --create-home --home-dir /home/appuser appuser \
    && mkdir -p /app/config-store /app/data/chroma \
    && chown -R appuser:appuser /app

WORKDIR /app

# venv + baked caches only — uv and builder apt crumbs stay out of the final image.
COPY --from=python-builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=python-builder --chown=appuser:appuser /app/.cache /app/.cache
COPY --from=python-builder --chown=appuser:appuser /app/.tesseract-rs /app/.tesseract-rs

# Flatten backend/ into /app so uvicorn app:app and relative paths (config/, .env) stay unchanged.
COPY --chown=appuser:appuser backend/ .
# Console at /ui/ always comes from the web-builder stage, never a stale local dist/.
COPY --from=web-builder --chown=appuser:appuser /frontend/dist ./frontend/dist

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
