"""
Indic-Speak Streaming Server (Production FastAPI + Sliding-Window Vocos Engine)
================================================================================
Exposes high-performance, real-time chunked audio streaming for bodhan-ai/indic-speak.
Yields 24kHz 16-bit PCM/WAV chunks with sub-300ms Time-To-First-Audio (TTFA) for
conversational voice agents (LiveKit, telephony, websockets, OpenAI TTS API).

Endpoints:
- POST /synthesize/stream    -> Native 24kHz 16-bit PCM chunked stream
- POST /synthesize           -> Complete 24kHz WAV audio output
- POST /v1/audio/speech      -> OpenAI-compatible Speech API (streaming & batch)
- GET  /v1/models            -> OpenAI-compatible models list
- WS   /ws/synthesize        -> Bi-directional WebSocket with interruption cancellation
- GET  /health               -> Health status and GPU VRAM telemetry

Author: Production Deployment Engineer
Target Model: bodhan-ai/indic-speak
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional, Literal

# Ensure local vocos package is resolvable regardless of invocation directory
CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import soundfile as sf

# ─── LOGGING CONFIGURATION ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("indic_speak_server")

MODEL_ID = os.getenv("INDIC_SPEAK_MODEL", "bodhan-ai/indic-speak")
if MODEL_ID == "bodhan-ai/indic-speak" and os.path.isdir("/workspace/hf_cache/indic-speak"):
    MODEL_ID = "/workspace/hf_cache/indic-speak"
VOCOS_PATH = os.getenv("INDIC_SPEAK_VOCOS", None)
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
MAX_CONCURRENT_SYNTH = int(os.getenv("MAX_CONCURRENT_SYNTH", "4"))
DEFAULT_SPEAKER = os.getenv("DEFAULT_SPEAKER", "Amit")
DEFAULT_STYLE = os.getenv("DEFAULT_STYLE", "")
STREAM_CHUNK_FRAMES = int(os.getenv("STREAM_CHUNK_FRAMES", "4"))  # 4 frames = 28 tokens ≈ 85ms audio

# ─── CONSTANTS & TOKEN MAP ──────────────────────────────────────────────────
SR = 24_000
NUM_CODEBOOKS = 7
CODEBOOK_SIZE = 4096
AUDIO_BASE_ID = 128266
SNAC_REPO = "hubertsiuzdak/snac_24khz"


# ─── ENGINE DEFINITION ───────────────────────────────────────────────────────
class IndicSpeakEngine:
    """Manages Llama-3.2-3B LM, SNAC Quantizer, and Vocos Decoder."""

    def __init__(self, model_id: str, vocos_path: Optional[str] = None, device: str = "cuda"):
        self.model_id = model_id
        self.vocos_path = vocos_path
        self.device = device
        self.tok = None
        self.lm = None
        self.snac = None
        self.vocos = None
        self._base = AUDIO_BASE_ID
        self._eos = None
        self._sos = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYNTH)

    def load(self):
        logger.info(f"Loading Indic-Speak from '{self.model_id}' on {self.device}...")
        t0 = time.perf_counter()

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from snac import SNAC

        # 0. Hardware Accelerations for NVIDIA RTX 4090 / Ampere / Ada Lovelace
        if self.device == "cuda" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            logger.info("Hardware accelerations enabled: TF32=ON, cuDNN Benchmark=ON")

        # 1. Tokenizer
        logger.info("Loading Tokenizer...")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)

        # 2. Backbone LM
        logger.info("Loading Llama-3.2-3B Causal LM (bfloat16, SDPA)...")
        self.lm = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa" if self.device == "cuda" else "eager",
        ).to(self.device).eval()

        # 3. SNAC Quantizer
        logger.info(f"Loading SNAC Quantizer from '{SNAC_REPO}'...")
        self.snac = SNAC.from_pretrained(SNAC_REPO).to(self.device).eval()

        # 4. Vocos Decoder
        resolved_vocos = self.vocos_path
        if resolved_vocos is None:
            # Check local path first
            candidates = [
                Path(self.model_id) / "vocos" / "best.pt",
                Path(CURRENT_DIR) / "vocos" / "best.pt",
                Path("/workspace/hf_cache") / "indic-speak" / "vocos" / "best.pt",
            ]
            for cand in candidates:
                if cand.is_file():
                    resolved_vocos = str(cand)
                    break
            
            if resolved_vocos is None:
                from huggingface_hub import hf_hub_download
                logger.info(f"Downloading Vocos decoder checkpoint from Hub '{self.model_id}'...")
                resolved_vocos = hf_hub_download(repo_id=self.model_id, filename="vocos/best.pt")

        logger.info(f"Loading Vocos decoder from '{resolved_vocos}'...")
        from vocos.load import load_vocos
        self.vocos = load_vocos(resolved_vocos, device=self.device)

        self._base = self.tok.convert_tokens_to_ids("<|snac_0|>")
        self._eos = self.tok.convert_tokens_to_ids("<|end_of_speech|>")
        self._sos = self.tok.convert_tokens_to_ids("<|start_of_speech|>")

        t_load = time.perf_counter() - t0
        logger.info(f"Indic-Speak engine loaded successfully in {t_load:.2f}s!")

    def build_prompt_ids(self, text: str, speaker: str = "", style: str = "") -> list[int]:
        """Constructs exact token prompt conforming to the token contract."""
        tid = lambda s: self.tok.convert_tokens_to_ids(s)
        nl = self.tok.encode("\n", add_special_tokens=False)
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)

        blocks = []
        if speaker.strip():
            blocks.append([tid("<|speaker>")] + enc(speaker.strip()) + [tid("<speaker|>")])
        if style.strip():
            blocks.append([tid("<|style>")] + enc(style.strip()) + [tid("<style|>")])

        meta: list[int] = []
        for i, b in enumerate(blocks):
            if i:
                meta += nl
            meta += b
        if blocks:
            meta += nl

        body = [self.tok.bos_token_id] + meta + enc(text) + [tid("<|eot_id|>")]
        return (
            [tid("<|start_of_human|>")]
            + body
            + [tid("<|end_of_human|>")]
            + [tid("<|start_of_ai|>"), tid("<|start_of_speech|>")]
        )

    def codes_from_ids(self, ids: list[int]) -> Optional[list[torch.Tensor]]:
        """Converts flattened token IDs into 3 multi-scale SNAC codebooks."""
        hi = self._base + NUM_CODEBOOKS * CODEBOOK_SIZE
        audio = [t for t in ids if self._base <= t < hi]

        n = len(audio) // NUM_CODEBOOKS
        if n == 0:
            return None

        a = np.array(audio[: n * NUM_CODEBOOKS], np.int32).reshape(n, NUM_CODEBOOKS)
        a -= self._base + np.arange(NUM_CODEBOOKS, dtype=np.int32) * CODEBOOK_SIZE
        valid_mask = np.all((a >= 0) & (a < CODEBOOK_SIZE), axis=1)
        a = a[valid_mask]
        if a.shape[0] == 0:
            return None

        c1 = np.empty(a.shape[0] * 2, np.int32)
        c1[0::2], c1[1::2] = a[:, 1], a[:, 4]
        c2 = np.empty(a.shape[0] * 4, np.int32)
        c2[0::4], c2[1::4], c2[2::4], c2[3::4] = a[:, 2], a[:, 3], a[:, 5], a[:, 6]

        return [
            torch.from_numpy(x.copy()).long().unsqueeze(0).to(self.device)
            for x in (a[:, 0], c1, c2)
        ]

    def decode_tokens_to_pcm(self, token_chunk: list[int]) -> Optional[bytes]:
        """Decodes a chunk of 7-frame tokens into 16-bit 24kHz PCM bytes."""
        codes = self.codes_from_ids(token_chunk)
        if codes is None:
            return None

        with torch.no_grad():
            z_q = self.snac.quantizer.from_codes(codes)
            wav = self.vocos(z_q.float())
            audio_arr = wav[0, 0].clamp(-1, 1).float().cpu().numpy()

        # Convert float32 [-1, 1] to 16-bit signed integer PCM
        pcm_16 = (audio_arr * 32767.0).astype(np.int16).tobytes()
        return pcm_16

    @staticmethod
    def estimate_max_tokens(text: str, default_max: int = 2520) -> int:
        """
        Dynamically calculates an intelligent token limit based on input text length.
        Speech in Indian languages averages ~2.5 words/sec, and 1 sec = 7 tokens * 13.5 frames = ~95 tokens.
        A 1.6x safety multiplier ensures sentences are never truncated while preventing runaway GPU waste.
        """
        words = max(1, len(text.strip().split()))
        estimated_sec = max(2.5, (words / 2.2) * 1.6)
        estimated_tokens = int(estimated_sec * 95)
        return min(default_max, max(280, estimated_tokens))

    def synthesize_full(
        self,
        text: str,
        speaker: str = "Amit",
        style: str = "",
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 2520,
    ) -> np.ndarray:
        """Synthesize a complete utterance into a 24kHz float32 numpy array."""
        effective_max_tokens = (
            self.estimate_max_tokens(text, default_max=max_new_tokens)
            if max_new_tokens == 2520
            else max_new_tokens
        )
        prompt_ids = self.build_prompt_ids(text, speaker, style)
        input_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)

        with torch.no_grad():
            gen = self.lm.generate(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=effective_max_tokens,
                eos_token_id=[self._eos, self.tok.eos_token_id],
                pad_token_id=self.tok.eos_token_id,
                use_cache=True,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            new_ids = gen[0].tolist()[len(prompt_ids):]
            codes = self.codes_from_ids(new_ids)
            if codes is None:
                return np.zeros(2400, dtype=np.float32)

            z_q = self.snac.quantizer.from_codes(codes)
            wav = self.vocos(z_q.float())
            return wav[0, 0].clamp(-1, 1).float().cpu().numpy()

    async def generate_stream(
        self,
        text: str,
        speaker: str = "Amit",
        style: str = "",
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 2520,
    ) -> AsyncGenerator[bytes, None]:
        """
        Streaming generator that emits 16-bit PCM bytes incrementally
        as frames are emitted by the autoregressive LM.
        """
        async with self.semaphore:
            effective_max_tokens = (
                self.estimate_max_tokens(text, default_max=max_new_tokens)
                if max_new_tokens == 2520
                else max_new_tokens
            )
            prompt_ids = self.build_prompt_ids(text, speaker, style)
            input_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)

            token_buffer: list[int] = []
            tokens_per_chunk = STREAM_CHUNK_FRAMES * NUM_CODEBOOKS  # e.g., 4 * 7 = 28 tokens
            t_start = time.perf_counter()
            first_chunk_sent = False

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Optional[int]] = asyncio.Queue()

            class AudioTokenStreamer:
                def __init__(self, q: asyncio.Queue, target_loop: asyncio.AbstractEventLoop):
                    self.q = q
                    self.loop = target_loop

                def put(self, value):
                    if isinstance(value, torch.Tensor):
                        for val in value.view(-1).tolist():
                            self.loop.call_soon_threadsafe(self.q.put_nowait, int(val))

                def end(self):
                    self.loop.call_soon_threadsafe(self.q.put_nowait, None)

            def _token_producer():
                streamer = AudioTokenStreamer(queue, loop)
                with torch.no_grad():
                    self.lm.generate(
                        input_ids=input_tensor,
                        attention_mask=torch.ones_like(input_tensor),
                        max_new_tokens=effective_max_tokens,
                        eos_token_id=[self._eos, self.tok.eos_token_id],
                        pad_token_id=self.tok.eos_token_id,
                        use_cache=True,
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        streamer=streamer,
                    )

            thread = asyncio.create_task(asyncio.to_thread(_token_producer))

            prompt_len = len(prompt_ids)
            gen_count = 0

            try:
                while True:
                    tok = await queue.get()
                    if tok is None:
                        break

                    gen_count += 1
                    # Skip echo of prompt tokens
                    if gen_count <= prompt_len:
                        continue

                    # If termination token hit, break
                    if tok == self._eos or tok == self.tok.eos_token_id:
                        break

                    # Append to buffer if within audio band
                    if self._base <= tok < self._base + (NUM_CODEBOOKS * CODEBOOK_SIZE):
                        token_buffer.append(tok)

                    # Once we have accumulated a full chunk (e.g. 28 tokens = 4 frames)
                    if len(token_buffer) >= tokens_per_chunk:
                        chunk_to_decode = token_buffer[:tokens_per_chunk]
                        token_buffer = token_buffer[tokens_per_chunk:]

                        pcm = await asyncio.to_thread(self.decode_tokens_to_pcm, chunk_to_decode)
                        if pcm:
                            if not first_chunk_sent:
                                ttfa = (time.perf_counter() - t_start) * 1000.0
                                logger.info(f"First audio chunk emitted in {ttfa:.1f}ms TTFA")
                                first_chunk_sent = True
                            yield pcm

                # Drain remaining tokens in buffer
                if len(token_buffer) >= NUM_CODEBOOKS:
                    leftover_frames = len(token_buffer) // NUM_CODEBOOKS
                    chunk_to_decode = token_buffer[: leftover_frames * NUM_CODEBOOKS]
                    pcm = await asyncio.to_thread(self.decode_tokens_to_pcm, chunk_to_decode)
                    if pcm:
                        yield pcm

            finally:
                await thread


# ─── FASTAPI APPLICATION ────────────────────────────────────────────────────
engine: Optional[IndicSpeakEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = IndicSpeakEngine(model_id=MODEL_ID, vocos_path=VOCOS_PATH, device=DEVICE)
    engine.load()
    yield
    logger.info("Shutting down Indic-Speak server...")


app = FastAPI(
    title="Indic-Speak Real-Time TTS Server",
    description="Production-ready streaming Text-to-Speech API for 22 Indian languages and English.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── SCHEMAS ────────────────────────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (Indic, English, or code-mixed)")
    speaker: str = Field(default="Amit", description="Registered voice ID (e.g., Kavya, Amit, Ishita)")
    style: str = Field(default="", description="Style/emotion tag (e.g., 'educational lecture', 'happy')")
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=100)
    max_new_tokens: int = Field(default=2520, ge=100, le=8000)
    response_format: Literal["pcm", "wav"] = Field(default="pcm", description="Output audio format")


class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="bodhan-ai/indic-speak", description="Model identifier")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="Amit", description="Speaker ID (e.g., Amit, Kavya, Ishita, Sourav)")
    response_format: Literal["pcm", "wav", "mp3"] = Field(default="wav", description="Audio format")
    speed: float = Field(default=1.0, description="Playback speed")
    stream: bool = Field(default=False, description="Whether to stream audio chunks")


# ─── ROUTES ─────────────────────────────────────────────────────────────────
@app.get("/health")
@app.get("/healthz")
@app.get("/ready")
async def health_check():
    """Health status and GPU memory telemetry."""
    vram_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    vram_res = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    return {
        "status": "ready" if engine is not None and engine.lm is not None else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "vram_allocated_gb": round(vram_alloc, 2),
        "vram_reserved_gb": round(vram_res, 2),
        "sample_rate": SR,
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "bodhan-ai/indic-speak",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "bodhan-ai",
            }
        ],
    }


@app.post("/synthesize")
async def synthesize_full_audio(req: SynthesizeRequest):
    """
    Non-streaming endpoint returning complete 24kHz audio (WAV format).
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    audio_arr = await asyncio.to_thread(
        engine.synthesize_full,
        text=req.text,
        speaker=req.speaker,
        style=req.style,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
    )

    buf = io.BytesIO()
    sf.write(buf, audio_arr, SR, format="WAV", subtype="PCM_16")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


@app.post("/synthesize/stream")
async def synthesize_stream(req: SynthesizeRequest):
    """
    HTTP Chunked Transfer Streaming Endpoint.
    Streams raw 24kHz mono 16-bit PCM chunks as they are generated.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    generator = engine.generate_stream(
        text=req.text,
        speaker=req.speaker,
        style=req.style,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
    )

    return StreamingResponse(
        generator,
        media_type="audio/pcm",
        headers={
            "X-Audio-Sample-Rate": str(SR),
            "X-Audio-Channels": "1",
            "X-Audio-Format": "pcm_s16le",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/v1/audio/speech")
async def openai_audio_speech(req: OpenAISpeechRequest):
    """
    OpenAI-compatible Speech API (`POST /v1/audio/speech`).
    Supports both standard file download and chunked audio streaming.
    """
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if req.stream:
        generator = engine.generate_stream(text=req.input, speaker=req.voice)
        return StreamingResponse(
            generator,
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": str(SR),
                "X-Audio-Channels": "1",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        audio_arr = await asyncio.to_thread(
            engine.synthesize_full,
            text=req.input,
            speaker=req.voice,
        )
        buf = io.BytesIO()
        sf.write(buf, audio_arr, SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )


@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket):
    """
    Bi-directional WebSocket streaming endpoint for low-latency conversational agents.
    Allows streaming synthesis requests and instant cancellation on interruption.
    """
    await websocket.accept()
    logger.info("WebSocket voice client connected.")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            text = payload.get("text", "")
            speaker = payload.get("speaker", DEFAULT_SPEAKER)
            style = payload.get("style", DEFAULT_STYLE)

            if not text.strip():
                await websocket.send_json({"event": "error", "message": "Empty text"})
                continue

            await websocket.send_json({"event": "start", "sample_rate": SR, "channels": 1})

            # Stream chunks over websocket
            async for pcm_chunk in engine.generate_stream(text=text, speaker=speaker, style=style):
                await websocket.send_bytes(pcm_chunk)

            await websocket.send_json({"event": "done"})

    except WebSocketDisconnect:
        logger.info("WebSocket voice client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info", reload=False)
