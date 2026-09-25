# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ARG INSTALL_EXPLAINABILITY=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Pillow wheels only need these small runtime libraries. Install them before
# switching away from root so the image does not ship apt metadata or caches.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Install only the hosted-inference backend set. CUDA-local PyTorch pins and
# research dependencies are deliberately excluded from the production image.
COPY backend/requirements.in backend/requirements.txt backend/constraints.txt ./backend/
RUN python -m pip install --no-cache-dir -r backend/requirements.txt

# Optional Grad-CAM dependencies are opt-in. The base hosted-inference image
# stays small and does not pull a CUDA runtime into production. The CPU wheel
# backend is used for the supported optional image; use a separate CUDA build
# environment when GPU inference is required.
COPY backend/requirements-explainability.in backend/requirements-explainability.txt backend/requirements-explainability.lock ./backend/
RUN if [ "$INSTALL_EXPLAINABILITY" = "true" ]; then \
      python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r backend/requirements-explainability.txt; \
    fi

# Fixed IDs make ownership and read-only mount behavior predictable across
# image rebuilds. Create the user before copying application files so
# COPY --chown can resolve the account.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /nonexistent \
       --shell /usr/sbin/nologin app

# Application code is root-owned and read-only; the service user can execute
# and import it but cannot modify or inject code into /app.
COPY --chown=root:root backend/ ./backend/
COPY --chown=root:root src/ ./src/
RUN chown -R root:root /app \
    && chmod -R a-w /app

# /tmp is writable for runtime tooling, while /app remains immutable to the
# service process. Optional checkpoints are mounted by operators rather than
# copied into the image.
RUN mkdir -p /tmp

USER 10001:10001

EXPOSE 8000

# /health reports process liveness even when hosted-model configuration is not
# ready. The shell wrapper also avoids relying on curl or extra image packages.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# Keep the existing single-worker deployment contract and endpoints unchanged.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
