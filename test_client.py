#!/usr/bin/env python3
"""
Indic-Speak Test Client
=======================
Send text synthesis requests to the Indic-Speak server and save the generated 24kHz audio.
Supports configurable base URLs (RunPod HTTP proxy, SSH tunnel, or local server).

Usage Examples:
    # 1. Health check & GPU info:
    python test_client.py --url http://localhost:8000 --health

    # 2. List available studio voices and languages:
    python test_client.py --url http://localhost:8000 --voices

    # 3. Generate Hindi speech via OpenAI-compatible endpoint:
    python test_client.py --url http://localhost:8000 --text "नमस्ते! आपका स्वागत है।" --speaker Kavya --lang hi-IN --out hindi.wav

    # 4. Stream real-time Tamil audio and measure Time-to-First-Audio (TTFA):
    python test_client.py --url https://<POD-ID>-8000.proxy.runpod.net --stream --text "வணக்கம், உங்கள் ஆர்டர் தயாராக உள்ளது." --lang ta-IN --out tamil.wav

    # 5. English technical speech:
    python test_client.py --text "Deep learning accelerates speech generation across Indian languages." --speaker Aarav --lang en-IN --out english.wav
"""

import argparse
import json
import os
import sys
import time
import wave
import urllib.request
import urllib.error

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Default Base URL can be overridden by environment variable or CLI argument
DEFAULT_BASE_URL = os.getenv("INDIC_SPEAK_URL", os.getenv("BASE_URL", "http://localhost:8000"))
SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # 16-bit mono PCM = 48,000 bytes/sec


def check_health(base_url: str):
    """Query /health endpoint to verify server readiness and GPU memory."""
    url = f"{base_url}/health"
    print(f"[*] Checking health at: {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IndicSpeakClient/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("[+] Server is HEALTHY and READY!")
            print(f"    - Model:    {data.get('model')}")
            print(f"    - Device:   {data.get('device')} ({data.get('gpu_name', 'N/A')})")
            vram = data.get('vram_allocated_gb', data.get('gpu_memory_allocated_gb', 'N/A'))
            print(f"    - VRAM:     {vram} GB allocated")
            print(f"    - Voices:   {data.get('supported_voices')} studio speakers")
            print(f"    - Langs:    {data.get('supported_languages')} Indic languages")
            return True
    except Exception as e:
        print(f"[-] Health check failed: {e}")
        print("    Ensure your RunPod pod is running and port 8000 is open/proxied.")
        return False


def list_voices(base_url: str):
    """List available voices from /voices endpoint."""
    url = f"{base_url}/voices"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IndicSpeakClient/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"\n[+] Total Studio Voices: {len(data.get('voices', []))}")
            print(f"    Sample Voices: {', '.join(data.get('voices', [])[:12])}...")
            print(f"[+] Total Languages: {len(data.get('languages', []))}")
            print(f"    Supported:     {', '.join(data.get('languages', []))}\n")
    except Exception as e:
        print(f"[-] Failed to fetch voices: {e}")


def write_pcm_to_wav(pcm_data: bytes, output_path: str, sample_rate: int = SAMPLE_RATE):
    """Encapsulates raw 16-bit mono PCM bytes with a standard 44-byte WAV header."""
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)     # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


def synthesize_openai(base_url: str, text: str, voice: str, language: str, output_path: str):
    """
    Synthesize audio using the OpenAI-compatible /v1/audio/speech endpoint.
    Returns standard 24kHz WAV audio stream.
    """
    url = f"{base_url}/v1/audio/speech"
    payload = {
        "input": text,
        "voice": voice,
        "model": language,
        "response_format": "wav"
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json", "User-Agent": "IndicSpeakClient/1.0"}
    )

    print(f"\n[*] Sending request to: {url}")
    print(f"    Text:     '{text}'")
    print(f"    Voice:    {voice} | Lang: {language}")
    print(f"    Endpoint: OpenAI-compatible (/v1/audio/speech)")

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            audio_bytes = resp.read()
            total_latency = time.time() - start_time

            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            file_size_kb = len(audio_bytes) / 1024
            # 24kHz 16-bit mono WAV has a 44-byte header
            pcm_len = max(0, len(audio_bytes) - 44)
            audio_duration = pcm_len / BYTES_PER_SECOND
            rtf = total_latency / audio_duration if audio_duration > 0 else 0

            print("\n" + "=" * 50)
            print("🎉 Audio Generated Successfully!")
            print(f"    - Output File:    {output_path}")
            print(f"    - File Size:      {file_size_kb:.1f} KB")
            print(f"    - Audio Duration: {audio_duration:.2f} s")
            print(f"    - Total Latency:  {total_latency * 1000:.1f} ms")
            print(f"    - RTF (Speed):    {rtf:.3f} ({1/rtf:.1f}x real-time)" if rtf > 0 else "")
            print("=" * 50)
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        print(f"[-] HTTP Error {e.code}: {error_msg}")
    except Exception as e:
        print(f"[-] Request failed: {e}")


def synthesize_stream(base_url: str, text: str, speaker: str, language: str, output_path: str, chunk_frames: int = 4):
    """
    Synthesize audio using the low-latency /synthesize/stream endpoint.
    Measures Time-to-First-Audio (TTFA) and streams raw PCM chunks.
    """
    url = f"{base_url}/synthesize/stream"
    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "chunk_frames": chunk_frames
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json", "User-Agent": "IndicSpeakClient/1.0"}
    )

    print(f"\n[*] Sending streaming request to: {url}")
    print(f"    Text:         '{text}'")
    print(f"    Speaker:      {speaker} | Lang: {language}")
    print(f"    Chunk Frames: {chunk_frames} (~85ms audio / chunk)")

    start_time = time.time()
    ttfa = None
    chunks_received = 0
    all_pcm_bytes = bytearray()

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            while True:
                chunk = resp.read(1920)  # 1920 bytes = 40ms of 24kHz 16-bit PCM
                if not chunk:
                    break

                if ttfa is None:
                    ttfa = time.time() - start_time
                    print(f"⚡ Time-to-First-Audio (TTFA): {ttfa * 1000:.1f} ms!")

                chunks_received += 1
                all_pcm_bytes.extend(chunk)

            total_latency = time.time() - start_time
            audio_duration = len(all_pcm_bytes) / BYTES_PER_SECOND
            rtf = total_latency / audio_duration if audio_duration > 0 else 0

            # Save as valid WAV file
            write_pcm_to_wav(bytes(all_pcm_bytes), output_path)
            file_size_kb = os.path.getsize(output_path) / 1024

            print("\n" + "=" * 50)
            print("🎉 Stream Completed & Saved to WAV!")
            print(f"    - Output File:    {output_path}")
            print(f"    - TTFA (Latency): {ttfa * 1000:.1f} ms" if ttfa else "N/A")
            print(f"    - Chunks Read:    {chunks_received}")
            print(f"    - Total Bytes:    {len(all_pcm_bytes):,} bytes ({file_size_kb:.1f} KB)")
            print(f"    - Audio Duration: {audio_duration:.2f} s")
            print(f"    - Total Time:     {total_latency * 1000:.1f} ms")
            print(f"    - RTF (Speed):    {rtf:.3f} ({1/rtf:.1f}x real-time)" if rtf > 0 else "")
            print("=" * 50)
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8", errors="ignore")
        print(f"[-] HTTP Error {e.code}: {error_msg}")
    except Exception as e:
        print(f"[-] Streaming failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Indic-Speak TTS Test Client with configurable Base URL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of Indic-Speak server (default: {DEFAULT_BASE_URL} or $INDIC_SPEAK_URL)"
    )
    parser.add_argument(
        "--text",
        default="नमस्ते! आपका स्वागत है। यह इंडिक स्पीक का लाइव ऑडियो टेस्ट है।",
        help="Text string to synthesize"
    )
    parser.add_argument(
        "--speaker", "--voice",
        default="Kavya",
        help="Studio voice name (default: Kavya)"
    )
    parser.add_argument(
        "--lang", "--language",
        default="hi-IN",
        help="Language code (e.g. hi-IN, ta-IN, te-IN, en-IN, bn-IN, etc.)"
    )
    parser.add_argument(
        "--out", "--output",
        default="output.wav",
        help="Output audio filepath (default: output.wav)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use low-latency chunked PCM streaming endpoint (/synthesize/stream) instead of /v1/audio/speech"
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=4,
        help="Number of SNAC frames per chunk when streaming (default: 4)"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Perform health check on server and exit"
    )
    parser.add_argument(
        "--voices",
        action="store_true",
        help="List all supported voices/languages and exit"
    )

    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    if args.health:
        check_health(base_url)
        return

    if args.voices:
        list_voices(base_url)
        return

    # Run synthesis
    if args.stream:
        synthesize_stream(
            base_url=base_url,
            text=args.text,
            speaker=args.speaker,
            language=args.lang,
            output_path=args.out,
            chunk_frames=args.chunk_frames
        )
    else:
        synthesize_openai(
            base_url=base_url,
            text=args.text,
            voice=args.speaker,
            language=args.lang,
            output_path=args.out
        )


if __name__ == "__main__":
    main()
