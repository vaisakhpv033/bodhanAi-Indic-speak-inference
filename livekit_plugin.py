"""
Indic-Speak LiveKit Plugin — Real-Time Streaming Edition
=========================================================
Integrates bodhan-ai/indic-speak into the LiveKit Agents Voice Pipeline.
Streams raw 24kHz 16-bit PCM chunks directly into LiveKit's AudioEmitter,
achieving sub-300ms Time-To-First-Byte (TTFB) with instantaneous cancellation
on user speech interruption.

Target Framework: LiveKit Agents (Python)
Compatible Model: bodhan-ai/indic-speak (via Indic-Speak FastAPI Streaming Server)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Optional, Literal

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    DEFAULT_API_CONNECT_OPTIONS,
    tts,
    utils,
)

logger = logging.getLogger("indic_speak_livekit")

# ─── DEFAULT VOICE PRESETS BY LANGUAGE ──────────────────────────────────────
INDIC_SPEAK_VOICE_MAP = {
    # Hindi
    "hi": {"female": "Kavya", "male": "Amit"},
    "hi-in": {"female": "Kavya", "male": "Amit"},
    # English (Indian)
    "en": {"female": "Kavya", "male": "Amit"},
    "en-in": {"female": "Kavya", "male": "Amit"},
    # Bengali
    "bn": {"female": "Ishita", "male": "Sourav"},
    # Tamil
    "ta": {"female": "Anitha", "male": "Arun"},
    # Telugu
    "te": {"female": "Sravani", "male": "Vamsi"},
    # Kannada
    "kn": {"female": "Deepika", "male": "Adarsh"},
    # Malayalam
    "ml": {"female": "Lakshmi", "male": "Kiran"},
    # Marathi
    "mr": {"female": "Anagha", "male": "Chinmay"},
    # Gujarati
    "gu": {"female": "Dhara", "male": "Parth"},
    # Punjabi
    "pa": {"female": "Kaur", "male": "Manpreet"},
    # Odia
    "or": {"female": "Itishree", "male": "Akash"},
    # Assamese
    "as": {"female": "Prastuti", "male": "Ankur"},
    # Urdu
    "ur": {"female": "Saba", "male": "Zaid"},
    # Sanskrit
    "sa": {"female": "Bharati", "male": "Aryaman"},
    # Nepali
    "ne": {"female": "Srijana", "male": "Sagar"},
}


@dataclass
class _TTSOptions:
    server_url: str
    speaker: str
    style: str
    temperature: float
    top_p: float
    sample_rate: int


class IndicSpeakTTS(tts.TTS):
    """
    LiveKit TTS Plugin for Indic-Speak (Bodhan AI / AI4Bharat).
    """

    def __init__(
        self,
        *,
        server_url: Optional[str] = None,
        speaker: str = "Amit",
        language: str = "hi-IN",
        gender: Literal["male", "female"] = "male",
        style: str = "educational lecture",
        temperature: float = 0.6,
        top_p: float = 0.9,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=24_000,
            num_channels=1,
        )

        resolved_url = server_url or os.getenv("INDIC_SPEAK_SERVER_URL", "http://localhost:8000")
        resolved_url = resolved_url.rstrip("/")

        # Resolve voice if default name requested
        clean_lang = language.lower().replace("_", "-")
        resolved_speaker = speaker
        if clean_lang in INDIC_SPEAK_VOICE_MAP:
            resolved_speaker = INDIC_SPEAK_VOICE_MAP[clean_lang].get(gender, speaker)

        self._opts = _TTSOptions(
            server_url=resolved_url,
            speaker=resolved_speaker,
            style=style,
            temperature=temperature,
            top_p=top_p,
            sample_rate=24_000,
        )
        self._session = http_session

    @property
    def speaker(self) -> str:
        return self._opts.speaker

    def update_options(
        self,
        *,
        speaker: Optional[str] = None,
        style: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        if speaker is not None:
            self._opts.speaker = speaker
        if style is not None:
            self._opts.style = style
        if temperature is not None:
            self._opts.temperature = temperature

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> tts.ChunkedStream:
        return ChunkedStream(
            tts=self,
            text=text,
            opts=self._opts,
            session=self._ensure_session(),
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )


class ChunkedStream(tts.ChunkedStream):
    """
    Handles streaming HTTP chunked response and forwards PCM audio frames to LiveKit.
    Supports clean cancellation on user speech interruption.
    """

    def __init__(
        self,
        *,
        tts: IndicSpeakTTS,
        text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._text = text

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        url = f"{self._opts.server_url}/synthesize/stream"
        payload = {
            "text": self._text,
            "speaker": self._opts.speaker,
            "style": self._opts.style,
            "temperature": self._opts.temperature,
            "top_p": self._opts.top_p,
            "response_format": "pcm",
        }

        logger.debug(f"[{request_id}] Requesting Indic-Speak streaming TTS for speaker '{self._opts.speaker}'")

        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=45, connect=5),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise APIStatusError(
                        f"Indic-Speak server returned HTTP {resp.status}: {error_text}",
                        status_code=resp.status,
                    )

                # Process 16-bit PCM stream
                # Each sample is 2 bytes (16-bit mono)
                # 24,000 samples/sec -> 48,000 bytes/sec
                # We emit in ~20ms chunks (480 samples = 960 bytes)
                frame_bytes = 960
                byte_buffer = bytearray()

                async for chunk in resp.content.iter_any():
                    # Check for client cancellation (user interruption)
                    if asyncio.current_task().cancelled():
                        logger.info(f"[{request_id}] TTS generation cancelled by user interruption.")
                        break

                    if not chunk:
                        continue

                    byte_buffer.extend(chunk)

                    while len(byte_buffer) >= frame_bytes:
                        raw_frame = bytes(byte_buffer[:frame_bytes])
                        del byte_buffer[:frame_bytes]

                        # Emit synthesized audio frame to LiveKit emitter
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=utils.AudioFrame(
                                    data=raw_frame,
                                    sample_rate=self._opts.sample_rate,
                                    num_channels=1,
                                    samples_per_channel=len(raw_frame) // 2,
                                ),
                            )
                        )

                # Drain remaining partial buffer
                if byte_buffer:
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            frame=utils.AudioFrame(
                                data=bytes(byte_buffer),
                                sample_rate=self._opts.sample_rate,
                                num_channels=1,
                                samples_per_channel=len(byte_buffer) // 2,
                            ),
                        )
                    )

        except asyncio.CancelledError:
            logger.info(f"[{request_id}] TTS request cancelled.")
            raise
        except aiohttp.ClientConnectorError as e:
            raise APIConnectionError(f"Cannot connect to Indic-Speak server at {url}: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError(f"Indic-Speak synthesis timed out: {e}") from e
        except Exception as e:
            logger.error(f"[{request_id}] Error during synthesis: {e}", exc_info=True)
            raise APIStatusError(f"Unexpected error during Indic-Speak synthesis: {e}") from e
