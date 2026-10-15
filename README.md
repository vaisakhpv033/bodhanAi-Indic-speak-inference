# Indic-Speak Production Deployment & Research Suite

Official deployment suite, architectural research, and production code for [`bodhan-ai/indic-speak`](https://huggingface.co/bodhan-ai/indic-speak) by **Bodhan AI (IIT Madras)** and **AI4Bharat**.

---

## 📁 Repository Structure

| File | Type | Description |
| :--- | :---: | :--- |
| [`RUNPOD_TEMPLATE_GUIDE.md`](./RUNPOD_TEMPLATE_GUIDE.md) | **Cloud Infra** | **RunPod Template Guide**: One-line Docker start command, template config, GPU sizing, and proxy test commands. |
| [`runpod_start.sh`](./runpod_start.sh) | **Cloud Infra** | Production autonomous startup script for RunPod GPU pods with uv package management and weight caching. |
| [`deploy_to_runpod.py`](./deploy_to_runpod.py) | **Cloud Infra** | Automated script to push code and start service on an active RunPod instance over SSH. |
| [`01_MODEL_BRIEF.md`](./01_MODEL_BRIEF.md) | **Document A** | **One-Page Model Brief**: Executive summary, suitability verdict, capabilities, and risks. |
| [`02_TECHNICAL_LEARNING_GUIDE.md`](./02_TECHNICAL_LEARNING_GUIDE.md) | **Document B** | **Technical Learning Guide**: Deep dive into Llama-3.2-3B LM, SNAC 7-token interleaving, Vocos vocoder, and math. |
| [`03_PRODUCTION_DEPLOYMENT_RUNBOOK.md`](./03_PRODUCTION_DEPLOYMENT_RUNBOOK.md) | **Document C** | **Production Deployment Runbook**: Step-by-step setup, capacity planning, NGINX, LiveKit, and operational checklist. |
| [`server.py`](./server.py) | **Executable** | Production **FastAPI Streaming Server** supporting OpenAI `/v1/audio/speech`, native PCM stream, and WebSocket. |
| [`vocos/`](./vocos/) | **Package** | Self-contained ConvNeXt-1D + iSTFT vocoder package matching the exact fine-tuned checkpoint architecture. |
| [`livekit_plugin.py`](./livekit_plugin.py) | **Executable** | **LiveKit Agents TTS Plugin** with chunked PCM audio streaming and instant interruption handling. |
| [`benchmark.py`](./benchmark.py) | **Executable** | Real-world benchmark tool measuring **TTFA, RTF, throughput, and latency percentiles**. |
| [`generated_audios/`](./generated_audios/) | **Audio Demos** | Verified **24kHz WAV audio samples** generated live on NVIDIA RTX 4090 GPU (Hindi, STEM, Tamil, English). |
| [`Dockerfile`](./Dockerfile) | **Infrastructure** | Production CUDA-accelerated Docker container. |
| [`docker-compose.yml`](./docker-compose.yml) | **Infrastructure** | Single-command deployment with GPU passthrough and health monitoring. |

---

## 🚀 Quickstart Guides

### 1. Deploy via RunPod Template UI (Recommended for Cloud)
See [`RUNPOD_TEMPLATE_GUIDE.md`](./RUNPOD_TEMPLATE_GUIDE.md) for full details.
* **Image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
* **Expose HTTP Ports:** `8000`
* **Volume Mount:** `/workspace` (40 GB)
* **Start Command:**
  ```bash
  bash -c "mkdir -p /workspace/indicspeak /workspace/hf_cache && export HF_HOME=/workspace/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 && apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg curl git && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=\$HOME/.local/bin:\$PATH && uv pip install --system 'transformers>=5.0.0' 'snac>=0.1.0' soundfile 'fastapi>=0.115.0' 'uvicorn[standard]>=0.30.0' pydantic aiohttp websockets huggingface_hub && cd /workspace/indicspeak && uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 || sleep infinity"
  ```

### 2. Launch Locally with Docker Compose
```bash
export HF_TOKEN="your_huggingface_token"
docker compose up -d
docker compose logs -f
```

### 3. Run Benchmark Suite
```bash
python benchmark.py --url http://localhost:8000 --concurrency 1,2,4 --trials 3
```

### 4. Integrate with LiveKit Voice Assistant
```python
from livekit_plugin import IndicSpeakTTS

tts = IndicSpeakTTS(
    server_url="http://localhost:8000",
    speaker="Kavya",
    language="hi-IN",
    style="educational lecture"
)
```
