# 🎙️ Indic-Speak (Bodhan AI / AI4Bharat) — RunPod Template Deployment Guide

This guide provides the complete, production-verified setup to deploy **Indic-Speak** (`bodhan-ai/indic-speak`) on **RunPod GPU Pods** using **RunPod Templates** and an automated container **Docker Start Command**.

The deployed service provides:
1. **Low-Latency Chunked Streaming Endpoint** (`POST /synthesize/stream`): Yields 24kHz 16-bit PCM chunks with sub-250ms Time-To-First-Audio (TTFA).
2. **OpenAI-Compatible Speech API** (`POST /v1/audio/speech`): Drop-in replacement for OpenAI TTS clients and LiveKit agents.
3. **Bi-directional WebSocket** (`WS /ws/synthesize`): Full-duplex conversational audio streaming with instant user interruption cancellation.
4. **Health & Telemetry Endpoint** (`GET /health`): Real-time GPU VRAM and service readiness checks.

---

## 1. 🛠️ RunPod Template Configuration

In your [RunPod Console](https://console.runpod.io/templates), click **New Template** and enter the following settings:

| Template Field | Value | Rationale |
| :--- | :--- | :--- |
| **Template Name** | `Indic-Speak-Streaming-TTS` | Descriptive identifier |
| **Template Type** | **Pod** | Standard dedicated GPU instance |
| **Container Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Official RunPod image; pre-cached globally, boots in <15s, CUDA 12.4 + Python 3.11 pre-configured. |
| **Container Disk** | `30 GB` | Space for system dependencies, uv packages, and OS layers |
| **Volume Disk** | `40 GB` (or larger) | Persistent volume mounted at `/workspace` |
| **Volume Mount Path**| `/workspace` | **Crucial:** Caches the 7.1 GB model weights across pod reboots |
| **Expose HTTP Ports**| `8000` | Port for the FastAPI and OpenAI-compatible server |
| **Expose TCP Ports** | `22` | Direct SSH access for monitoring and maintenance |

---

## 2. 🚀 Container Docker Start Command

You can use either **Strategy A (GitHub Auto-Pull & Launch — Recommended)**, **Strategy B (Local Volume Launch)**, or **Strategy C (Inline Provisioning)**:

### 🌟 Strategy A (Recommended): Auto-Pull from GitHub & Launch
Push this folder to a GitHub repository, then paste this command into the RunPod Template **Docker Start Command** field:

```bash
bash -c "mkdir -p /workspace && cd /workspace && if [ ! -d 'indicspeak' ]; then git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git indicspeak; else cd indicspeak && git pull; fi && cd /workspace/indicspeak && chmod +x runpod_start.sh && bash runpod_start.sh || sleep infinity"
```
*(For private GitHub repositories, replace the URL with: `https://$GITHUB_TOKEN@github.com/<USERNAME>/<REPO>.git` and add `GITHUB_TOKEN` to Environment Variables).*

### Strategy B: Launch from Pre-Synced Volume
If the code is already on `/workspace/indicspeak`:
```bash
bash -c "cd /workspace/indicspeak && chmod +x runpod_start.sh && bash runpod_start.sh || sleep infinity"
```

### Strategy C: Self-Contained Inline Command (Zero-Touch Provisioning)
```bash
bash -c "mkdir -p /workspace/indicspeak /workspace/hf_cache && export HF_HOME=/workspace/hf_cache && apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg curl git && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=\$HOME/.local/bin:\$PATH && uv pip install --system torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124 && uv pip install --system 'transformers>=5.0.0' 'snac>=0.1.0' soundfile 'fastapi>=0.115.0' 'uvicorn[standard]>=0.30.0' pydantic aiohttp websockets huggingface_hub && cd /workspace/indicspeak && uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 || sleep infinity"
```

### 🔍 Anatomy of the Start Command:
1. `mkdir -p /workspace/indicspeak /workspace/hf_cache`: Ensures persistent storage directories exist.
2. `export HF_HOME=/workspace/hf_cache`: Ensures the 7.1 GB model weights are cached on the network volume and **never re-downloaded** on pod restart.
3. `torch==2.6.0 torchvision==0.21.0`: **Critical Verified Discovery:** `transformers >= 5.0.0` uses the `torch.accelerator` API which requires PyTorch 2.6.0.
4. `apt-get install -y libsndfile1 ffmpeg`: Installs necessary C audio libraries.
5. `curl -LsSf https://astral.sh/uv/install.sh | sh`: Uses **astral uv** to install all dependencies in seconds.
6. `cd /workspace/indicspeak && uvicorn server:app ...`: Starts the high-performance async streaming server.
7. `|| sleep infinity`: Prevents RunPod from entering an infinite reboot loop if an error occurs.

---

## 3. 🌐 Environment Variables

Configure these key-value pairs under **Environment Variables** in the template:

| Variable | Value | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | `hf_YourApprovedHuggingFaceToken` | **Required:** Access token approved for `bodhan-ai/indic-speak`. |
| `HF_HOME` | `/workspace/hf_cache` | Path for persistent model weight caching. |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enables multi-threaded Rust downloads for maximum bandwidth. |
| `MAX_CONCURRENT_SYNTH` | `4` | Max concurrent speech generations allowed simultaneously on 1 GPU. |
| `STREAM_CHUNK_FRAMES` | `4` | Number of SNAC frames per chunk (4 frames = 28 tokens ≈ 85ms audio). |
| `DEVICE` | `cuda` | Target compute device. |

---

## 4. 🖥️ Recommended GPU Hardware

| GPU Model | VRAM | RTF (Speed) | Max Concurrent Streams | Recommendation |
| :--- | :---: | :---: | :---: | :--- |
| **NVIDIA RTX 4090** | **24 GB** | **~0.18 – 0.22** | **4 – 6 streams** | ⭐ **Best Price/Performance** (~$0.44/hr) |
| **NVIDIA RTX 3090** | **24 GB** | **~0.28 – 0.35** | **3 – 4 streams** | Budget-friendly option (~$0.29/hr) |
| **NVIDIA L4** | **24 GB** | **~0.22 – 0.26** | **4 – 5 streams** | Modern Ada Lovelace enterprise card |
| **NVIDIA A10G** | **24 GB** | **~0.30 – 0.36** | **3 – 4 streams** | Standard cloud enterprise option |
| **NVIDIA A40 / A100** | **48/80 GB**| **~0.15 – 0.20** | **12+ streams** | High-throughput enterprise clusters |

> **VRAM Usage**: Base model + Vocos occupies **~8.8 GB VRAM**. Under 4 active concurrent streams, VRAM peaks at **~10.2 GB**, leaving over 13 GB of safe headroom on any 24 GB card.

---

## 5. 📡 Accessing & Verifying the Deployed Pod

When the pod status changes to **Running**, RunPod provisions an external URL:
`https://<POD_ID>-8000.proxy.runpod.net`

### 5.1 Step 1: Health Check
```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```
**Expected Response:**
```json
{
  "status": "ready",
  "model": "bodhan-ai/indic-speak",
  "device": "cuda",
  "vram_allocated_gb": 8.84,
  "vram_reserved_gb": 9.42,
  "sample_rate": 24000
}
```

### 5.2 Step 2: Test OpenAI-Compatible Speech Generation
You can use standard OpenAI API clients or curl:
```bash
curl -X POST "https://<POD_ID>-8000.proxy.runpod.net/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "bodhan-ai/indic-speak",
       "input": "नमस्ते, आपका स्वागत है।",
       "voice": "Amit",
       "response_format": "wav"
     }' \
     --output test_speech.wav
```

### 5.3 Step 3: Test Real-Time Chunked Streaming (`POST /synthesize/stream`)
```python
import aiohttp
import asyncio
import time

async def test_streaming(pod_url):
    url = f"{pod_url.rstrip('/')}/synthesize/stream"
    payload = {
        "text": "प्रकाश की चाल लगभग 3 * 10^8 m/s होती है।",
        "speaker": "Amit",
        "response_format": "pcm"
    }
    
    t0 = time.perf_counter()
    first_chunk = False
    total_bytes = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            async for chunk in resp.content.iter_any():
                if chunk:
                    if not first_chunk:
                        ttfa = (time.perf_counter() - t0) * 1000
                        print(f"⚡ Time-To-First-Audio (TTFA): {ttfa:.1f} ms")
                        first_chunk = True
                    total_bytes += len(chunk)

    duration_s = total_bytes / (24000 * 2)
    print(f"✅ Generated {duration_s:.2f}s of 24kHz audio in {(time.perf_counter()-t0):.2f}s")

# Replace with your actual pod URL
asyncio.run(test_streaming("https://<POD_ID>-8000.proxy.runpod.net"))
```

---

## 6. 🎙️ LiveKit Agents Voice Assistant Integration

In your LiveKit agent orchestrator, point `IndicSpeakTTS` directly to your RunPod proxy URL:

```python
from livekit_plugin import IndicSpeakTTS

tts = IndicSpeakTTS(
    server_url="https://<POD_ID>-8000.proxy.runpod.net",
    speaker="Kavya",             # Indian female voice
    language="hi-IN",            # Hindi & Hinglish
    style="educational lecture"
)
```

---

## 7. 🔄 One-Click Script Deployment via SSH (Alternative Method)

If you already have a running RunPod pod and want to sync your local code directly:

1. Copy the code to your pod:
   ```bash
   scp -P <SSH_PORT> -r E:\Brocamp\After_brocamp\model-deployments\bodhanai\indicspeak root@<RUNPOD_IP>:/workspace/
   ```
2. SSH into the pod:
   ```bash
   ssh -p <SSH_PORT> root@<RUNPOD_IP>
   ```
3. Run the startup script:
   ```bash
   bash /workspace/indicspeak/runpod_start.sh
   ```
