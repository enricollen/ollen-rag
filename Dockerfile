# Dockerfile — OpenShift-compatible (runs under arbitrary UID in root group)
FROM python:3.13-slim

# HOME/XDG_CACHE_HOME point at /app so libraries resolving ~ under OpenShift's
# arbitrary UID (no passwd entry) still get a group-writable directory
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface

# LibreOffice + ImageMagick required by liteparse for Office/image formats
RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-nogui imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# OpenShift assigns a random UID in group 0: make app dirs group-writable
RUN mkdir -p /app/.cache/huggingface \
    && chgrp -R 0 /app \
    && chmod -R g=u /app

USER 1001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
