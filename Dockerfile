# Production Dockerfile for SatQuery AI Backend (FastAPI + PyTorch + LoRA)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt ./requirements.txt
# Install PyTorch CPU wheels for lightweight container deployment (or standard GPU if building on CUDA host)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application backend code, adapter checkpoints, and sample assets
COPY app/ ./app/
COPY backend/ ./backend/
COPY training/ ./training/
COPY data_prep/ ./data_prep/
COPY outputs/ ./outputs/

# Create runtime directories for continuous active learning persistence
RUN mkdir -p outputs/memory_crops outputs/blip-lora-satellite-adapter

# Expose backend port (Render / Cloud Run automatically binds to PORT env)
EXPOSE 8000

# Healthcheck to ensure microservice readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
