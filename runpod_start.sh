#!/usr/bin/env bash
# ==============================================================================
# runpod_start.sh — Production Startup Script for Indic-Speak on RunPod GPU Pods
# ==============================================================================
# Compatible with: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Target Model: bodhan-ai/indic-speak
# ==============================================================================
set -e

echo "================================================================"
echo "🚀 Starting Indic-Speak Real-Time TTS Server on RunPod"
echo "================================================================"

WORKSPACE="/workspace"
APP_DIR="${WORKSPACE}/indicspeak"
export HF_HOME="${WORKSPACE}/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME"
mkdir -p "$APP_DIR"

if [ -f "${APP_DIR}/.env" ]; then
    echo "[ENV] Sourcing environment variables from ${APP_DIR}/.env..."
    set -a
    source "${APP_DIR}/.env"
    set +a
fi

# 1. System packages
echo "[1/5] Checking system audio libraries..."
if ! dpkg -s libsndfile1 ffmpeg >/dev/null 2>&1; then
    echo "  --> Installing libsndfile1 and ffmpeg..."
    apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg curl git >/dev/null 2>&1 || true
fi

# 2. Fast package manager (Astral uv)
echo "[2/5] Ensuring fast package manager (uv)..."
if ! command -v uv &> /dev/null && [ ! -f "$HOME/.local/bin/uv" ]; then
    echo "  --> Installing astral uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
fi
UV_BIN="$HOME/.local/bin/uv"
[ -f "$UV_BIN" ] || UV_BIN="uv"

# 3. Python dependencies
echo "[3/5] Verifying and installing Python dependencies..."
# Ensure PyTorch 2.6.0 with CUDA 12.4 is installed (required for Transformers 5.x torch.accelerator API)
if ! python3 -c "import torch; exit(0 if torch.__version__.startswith('2.6') and torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  --> Upgrading to PyTorch 2.6.0+cu124 and torchvision 0.21.0..."
    $UV_BIN pip install --system torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
fi

$UV_BIN pip install --system \
    "transformers>=5.0.0" \
    "snac>=0.1.0" \
    "soundfile>=0.12.1" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.9.0" \
    "aiohttp>=3.10.0" \
    "websockets>=13.0" \
    "huggingface_hub" \
    "numpy>=1.26.0,<2.0.0" \
    "tabulate>=0.9.0"

# 4. Clone or synchronize code if not already present in /workspace/indicspeak
cd "$APP_DIR"
if [ ! -f "server.py" ]; then
    echo "[4/5] Code not found in $APP_DIR. Downloading from repository / source..."
    # If HF repo contains server.py or if repository is cloned
    if [ -n "$REPO_URL" ]; then
        git clone "$REPO_URL" "$APP_DIR"
    fi
fi

# 5. Pre-warm / Download Model Weights to /workspace/hf_cache
echo "[5/5] Pre-warming Indic-Speak model weights to persistent volume ($HF_HOME)..."
python3 -c "
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get('INDIC_SPEAK_MODEL', 'bodhan-ai/indic-speak')
token = os.environ.get('HF_TOKEN')
if token: token = token.strip()
print(f'Downloading / verifying {model_id}...')
snapshot_download(
    repo_id=model_id,
    token=token,
    local_dir='/workspace/hf_cache/indic-speak',
    allow_patterns=['*.safetensors', '*.json', '*.md', 'vocos/*']
)
print('Model weights verified in /workspace/hf_cache/indic-speak.')
" || echo "Warning: Pre-download failed, server will load on demand."

echo "================================================================"
echo "⚡ Launching FastAPI Server on 0.0.0.0:8000..."
echo "================================================================"
exec uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
