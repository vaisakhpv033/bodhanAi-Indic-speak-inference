# Indic-Speak Production Inference Server on RunPod

[![GitHub Repository](https://img.shields.io/badge/GitHub-bodhanAi--Indic--speak--inference-blue?logo=github)](https://github.com/vaisakhpv033/bodhanAi-Indic-speak-inference)
[![Model](https://img.shields.io/badge/HuggingFace-bodhan--ai%2Findic--speak-yellow?logo=huggingface)](https://huggingface.co/bodhan-ai/indic-speak)
[![Architecture](https://img.shields.io/badge/Architecture-Llama--3.2--3B%20%2B%20SNAC%2024kHz%20%2B%20Vocos-blueviolet)](#)
[![Latency](https://img.shields.io/badge/TTFA-~790ms%20(RTX%204090)-success)](#)
[![License](https://img.shields.io/badge/License-Indic%20Open%20Model%20License%20v1.0-green)](#)

Production streaming server, LiveKit agent plugin, vocoder engine, and RunPod template deployment suite for **`bodhan-ai/indic-speak`** (23 Indic languages + English, 44 studio voices, 24kHz audio).

---

## 🚀 RunPod Template Configuration

To deploy on [RunPod.io](https://www.runpod.io/console/serverless/user/templates), create a new **Pod Template** with the following settings:

| Field | Value | Notes |
| :--- | :--- | :--- |
| **Template Name** | `indic-speak-tts-server` | Custom template identifier |
| **Container Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Official PyTorch base image |
| **Container Disk** | `40 GB` | Space for packages and system binaries |
| **Volume Disk** | `40 GB` | Mounted at `/workspace` for persistent model weights |
| **Volume Mount Path**| `/workspace` | Preserves `hf_cache` across pod stops/restarts |
| **Expose HTTP Ports**| `8000` | Exposes FastAPI REST & WebSocket streaming |
| **Expose TCP Ports** | `22` | Direct SSH and SFTP access |

### Environment Variables
Configure these in the **Environment Variables** section of your template:

```env
HF_TOKEN=your_approved_huggingface_token
HF_HOME=/workspace/hf_cache
HF_HUB_ENABLE_HF_TRANSFER=1
INDIC_SPEAK_MODEL=bodhan-ai/indic-speak
DEVICE=cuda
MAX_CONCURRENT_SYNTH=4
PORT=8000
HOST=0.0.0.0
```

> [!IMPORTANT]
> `bodhan-ai/indic-speak` is a gated model on Hugging Face. Ensure your `HF_TOKEN` has approved access before starting the pod.

---

## ⚡ Container Start Command

Paste this exact one-line command into the **Container Start Command** field of your RunPod template:

```bash
bash -c 'if [ ! -d "/workspace/indicspeak" ]; then git clone https://github.com/vaisakhpv033/bodhanAi-Indic-speak-inference.git /workspace/indicspeak; else cd /workspace/indicspeak && git pull; fi && cd /workspace/indicspeak && bash runpod_start.sh'
```

### What This Command Does:
1. Clones `https://github.com/vaisakhpv033/bodhanAi-Indic-speak-inference` into `/workspace/indicspeak` on first boot.
2. Runs `git pull` if the repository already exists on the persistent volume.
3. Automatically installs `libsndfile1`, `ffmpeg`, and `uv`.
4. Upgrades to `torch==2.6.0+cu124` (required for Transformers 5.x `torch.accelerator` support).
5. Pre-warms and caches model weights inside `/workspace/hf_cache/indic-speak`.
6. Launches the FastAPI streaming server with sliding-window multi-scale SNAC decoding on `0.0.0.0:8000`.

---

## 📡 Connecting to Your Pod

Once your pod boots, you can connect using either method:

### Method A: RunPod HTTP Proxy (No SSH required)
RunPod provides an SSL HTTP proxy URL for exposed port 8000:
```
https://<YOUR-POD-ID>-8000.proxy.runpod.net
```

### Method B: SSH Port Forwarding (Direct & Low Latency)
Forward port `8000` from the pod to your local machine:
```bash
ssh -L 8000:localhost:8000 root@<POD_IP> -p <SSH_PORT> -i ~/.ssh/id_ed25519
```
Now access the server locally at `http://localhost:8000`.

---

## 🎧 How to Send Inputs & Get Audio Outputs

### 1. Check Server Health & GPU VRAM
Verify the server is ready to accept synthesis requests:

```bash
curl -s http://localhost:8000/health | jq .
```
**Response:**
```json
{
  "status": "ok",
  "model": "bodhan-ai/indic-speak",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_allocated_gb": 6.44,
  "supported_voices": 44,
  "supported_languages": 24
}
```

---

### 2. OpenAI-Compatible Endpoint (`POST /v1/audio/speech`)
Drop-in replacement for OpenAI TTS API clients. Returns a playable **24kHz WAV audio stream**.

#### Using `curl` (Save to `.wav` file):
```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "नमस्ते! आपका ऑर्डर कन्फर्म हो गया है और डिलीवरी कल सुबह तक पहुँच जाएगी।",
    "voice": "Kavya",
    "model": "hi-IN",
    "response_format": "wav"
  }' --output hindi_delivery.wav
```

#### Using Python (`requests`):
```python
import requests

url = "http://localhost:8000/v1/audio/speech"
payload = {
    "input": "Quantum computing harnesses the laws of quantum mechanics to solve problems too complex for classical computers.",
    "voice": "Aarav",
    "model": "en-IN",
    "response_format": "wav"
}

response = requests.post(url, json=payload, stream=True)
with open("quantum_lecture.wav", "wb") as f:
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            f.write(chunk)

print("Saved output to quantum_lecture.wav (24kHz WAV)")
```

---

### 3. Chunked Real-Time PCM Streaming (`POST /synthesize/stream`)
Ultra-low first-audio latency (`TTFA ~790ms`). Emits raw 24kHz 16-bit mono PCM chunks as tokens are generated using a sliding-window vocoder.

#### Python Streaming Client:
```python
import requests

url = "http://localhost:8000/synthesize/stream"
payload = {
    "text": "வணக்கம்! நீங்கள் கேட்ட விவரங்கள் தயாராக உள்ளன.",
    "speaker": "Kavya",
    "language": "ta-IN",
    "style": "warm conversational",
    "chunk_frames": 4
}

with requests.post(url, json=payload, stream=True) as r:
    total_bytes = 0
    for chunk in r.iter_content(chunk_size=1920):  # 1920 bytes = 40ms of 24kHz PCM
        if chunk:
            total_bytes += len(chunk)
            # Send chunk directly to WebRTC audio track or audio player
            print(f"Received audio chunk: {len(chunk)} bytes (Total: {total_bytes})")
```

#### Play Stream Directly with `ffplay` / `mpv`:
```bash
curl -s -N -X POST "http://localhost:8000/synthesize/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "यह एक लाइव ऑडियो टेस्ट है।", "speaker": "Kavya", "language": "hi-IN"}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -
```

---

### 4. WebSocket Streaming (`ws://localhost:8000/ws/synthesize`)
Full-duplex WebSocket connection for conversational AI agents.

```python
import asyncio
import json
import websockets

async def stream_audio():
    uri = "ws://localhost:8000/ws/synthesize"
    async with websockets.connect(uri) as ws:
        request = {
            "text": "హలో! మీ అభ్యర్థన విజయవంతంగా పూర్తయింది.",
            "speaker": "Kavya",
            "language": "te-IN"
        }
        await ws.send(json.dumps(request))
        
        with open("telugu_output.pcm", "wb") as f:
            while True:
                data = await ws.recv()
                if isinstance(data, bytes):
                    f.write(data)
                elif isinstance(data, str):
                    msg = json.loads(data)
                    if msg.get("event") == "eos":
                        print("Audio stream finished.")
                        break

asyncio.run(stream_audio())
```

---

### 5. LiveKit Voice Agent Integration
Plug `bodhan-ai/indic-speak` directly into LiveKit Agents pipelines with full interruption support:

```python
from livekit_plugin import IndicSpeakTTS

tts = IndicSpeakTTS(
    server_url="http://localhost:8000",
    speaker="Kavya",
    language="hi-IN",
    style="friendly customer service",
    chunk_frames=4
)

# Use in your LiveKit VoicePipelineAgent:
# agent = VoicePipelineAgent(vad=vad, stt=stt, llm=llm, tts=tts)
```

---

## 📊 Benchmarking & Performance

Run the included benchmark suite against your deployed pod:

```bash
python benchmark.py --url http://localhost:8000 --concurrency 1,2,4 --trials 3
```

### Live Benchmark Results on NVIDIA RTX 4090 (24GB):
* **p50 TTFA (Time-to-First-Audio):** `793.6 ms`
* **Real-Time Factor (RTF):** `0.076` (~13x faster than real-time)
* **VRAM Footprint:** `6.44 GB` (fits comfortably on RTX 3090, 4090, A4000, A5000, A100)
* **Audio Sampling Rate:** `24,000 Hz (24kHz)`

---

## 🎙️ Available Speakers & Languages

Query all available voices dynamically:
```bash
curl -s http://localhost:8000/voices | jq .
```

* **Languages (24):** `hi-IN` (Hindi), `ta-IN` (Tamil), `te-IN` (Telugu), `bn-IN` (Bengali), `mr-IN` (Marathi), `gu-IN` (Gujarati), `kn-IN` (Kannada), `ml-IN` (Malayalam), `pa-IN` (Punjabi), `or-IN` (Odia), `as-IN` (Assamese), `en-IN` (English), and all official 8th Schedule Indian languages.
* **Studio Voices (44):** `Kavya`, `Aarav`, `Meera`, `Rohan`, `Pooja`, `Vikram`, `Ananya`, `Kabir`, etc.

---

## 📁 Repository Files

* [`server.py`](./server.py): FastAPI streaming server with multi-scale SNAC decoding.
* [`runpod_start.sh`](./runpod_start.sh): Autonomous initialization script for RunPod pods.
* [`livekit_plugin.py`](./livekit_plugin.py): Real-time LiveKit Agent plugin.
* [`vocos/`](./vocos/): ConvNeXt-1D neural vocoder engine.
* [`benchmark.py`](./benchmark.py): TTFA and concurrency benchmark suite.
* [`RUNPOD_TEMPLATE_GUIDE.md`](./RUNPOD_TEMPLATE_GUIDE.md): In-depth guide with diagrams and troubleshooting.
* [`01_MODEL_BRIEF.md`](./01_MODEL_BRIEF.md): Architecture evaluation and suitability report.
* [`02_TECHNICAL_LEARNING_GUIDE.md`](./02_TECHNICAL_LEARNING_GUIDE.md): Technical deep-dive on SNAC interleaving & Vocos math.
* [`03_PRODUCTION_DEPLOYMENT_RUNBOOK.md`](./03_PRODUCTION_DEPLOYMENT_RUNBOOK.md): Enterprise NGINX, capacity planning & SSL guide.
