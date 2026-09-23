# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

# Install system deps for Pillow (libjpeg, zlib) and cleaning
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
# Explainability (Grad-CAM) requires numpy + torch. Optional: install when
# LOCAL_MODEL_PATH is set. Keep the base image small by installing only when
# needed; for a full deployment that includes explainability, uncomment:
# RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu

# Application code
COPY backend/ /app/backend/
COPY src/ /app/src/

# Make src/ importable
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
