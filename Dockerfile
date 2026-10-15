# Production Dockerfile for Indic-Speak Real-Time TTS Server
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install essential system dependencies (libsndfile for audio I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "transformers>=5.0.0.dev0" \
    "snac>=0.1.0" \
    "soundfile>=0.12.1" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.9.0" \
    "huggingface_hub[hf_transfer]>=0.25.0" \
    "numpy>=1.26.0,<2.0.0" \
    "aiohttp>=3.10.0" \
    "tabulate>=0.9.0"

# Copy local application code
COPY server.py /app/server.py
COPY benchmark.py /app/benchmark.py

# Create directory for Vocos decoder and cache
RUN mkdir -p /app/vocos /root/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
