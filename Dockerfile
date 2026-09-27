# Production Dockerfile for SatQuery AI Backend (Hugging Face Spaces / Render / Cloud Run)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    PYTHONPATH=/app \
    HOME=/home/appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (standard for Hugging Face Spaces security & permissions)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/outputs/memory_crops /app/outputs/blip-lora-satellite-adapter && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt ./requirements.txt
# Install PyTorch CPU wheels for lightweight container deployment
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application backend code, adapter checkpoints, and sample assets with correct ownership
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser training/ ./training/
COPY --chown=appuser:appuser data_prep/ ./data_prep/
COPY --chown=appuser:appuser outputs/ ./outputs/

# Switch to non-root user
USER appuser

# Expose ports (7860 for Hugging Face Spaces, 8000 for local / custom)
EXPOSE 7860
EXPOSE 8000

# Start Uvicorn ASGI server (uses PORT env if set by host, else defaults to 7860)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
