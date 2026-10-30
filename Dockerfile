# Multi-stage build: the builder stage has uv and builds everything, the runtime
# stage only gets the finished virtual environment and the trained model.
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

# ---------- builder: install dependencies, train the model ----------
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first: this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Train inside the build; the build fails if the model is not good enough.
RUN /app/.venv/bin/python -m ml_service.train --out /app/models/model.joblib --min-accuracy 0.9

# ---------- runtime: small image, non-root user ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
RUN groupadd --system app && useradd --system --gid app --no-create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/models /app/models
ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "ml_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
