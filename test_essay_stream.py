#!/usr/bin/env python3
"""
Indic-Speak Sentence-Delimited Essay Streaming Benchmark
=======================================================
Demonstrates real-time essay / paragraph streaming by segmenting text
into sentences based on punctuation delimiters (।, ., ?, !).
Simulates an LLM voice pipeline feeding sequential sentences to the TTS server.

Usage:
    # 1. Direct GPU / localhost:
    python test_essay_stream.py --url http://127.0.0.1:8000

    # 2. Remote RunPod Pod (over TCP or Proxy):
    python test_essay_stream.py --url https://<POD-ID>-8000.proxy.runpod.net --speaker Kavya --lang hi-IN --out essay.wav
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import wave
from typing import List, Dict, Any

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_BASE_URL = os.getenv("INDIC_SPEAK_URL", os.getenv("BASE_URL", "http://localhost:8000"))
SR = 24000
BYTES_PER_SEC = SR * 2  # 16-bit mono PCM = 48,000 bytes/sec

SAMPLE_HINDI_ESSAY = (
    "कृत्रिम बुद्धिमत्ता आधुनिक युग की सबसे क्रांतिकारी तकनीकों में से एक है। "
    "यह तकनीक मशीनों को इंसानों की तरह सोचने और सीखने की क्षमता प्रदान करती है। "
    "आज स्वास्थ्य, शिक्षा और उद्योग के हर क्षेत्र में इसका उपयोग तेजी से बढ़ रहा है। "
    "भविष्य में यह तकनीक हमारे काम करने के तरीके को पूरी तरह से बदल देगी।"
)

SAMPLE_ENGLISH_ESSAY = (
    "Artificial intelligence is transforming modern technology at an unprecedented pace. "
    "Machine learning models can now generate human-like speech and understand complex languages. "
    "As neural architectures advance, conversational agents will become an essential part of daily communication. "
    "The future of voice technology lies in seamless low-latency interaction."
)


def split_sentences(text: str) -> List[str]:
    """Splits a multi-sentence essay using Indic and Latin sentence boundaries."""
    # Delimit by Devanagari danda (।), period (.), question mark (?), exclamation (!)
    chunks = re.split(r"(?<=[।\.?!])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def stream_sentence(base_url: str, text: str, speaker: str, lang: str, chunk_frames: int = 4) -> Dict[str, Any]:
    """Streams a single sentence from the server and measures TTFA and latency."""
    url = f"{base_url}/synthesize/stream"
    payload = json.dumps({
        "text": text,
        "speaker": speaker,
        "language": lang,
        "chunk_frames": chunk_frames
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "IndicSpeakEssayClient/1.0"}
    )

    t_start = time.perf_counter()
    ttfa = None
    all_pcm = bytearray()
    chunks = 0

    with urllib.request.urlopen(req, timeout=90) as resp:
        while True:
            chunk = resp.read(1920)
            if not chunk:
                break
            if ttfa is None:
                ttfa = time.perf_counter() - t_start
            chunks += 1
            all_pcm.extend(chunk)

    t_total = time.perf_counter() - t_start
    audio_dur = len(all_pcm) / BYTES_PER_SEC
    rtf = t_total / audio_dur if audio_dur > 0 else 0

    return {
        "text": text,
        "pcm": bytes(all_pcm),
        "ttfa_ms": (ttfa * 1000) if ttfa else 0,
        "gen_sec": t_total,
        "audio_dur_sec": audio_dur,
        "rtf": rtf,
        "chunks": chunks
    }


def main():
    parser = argparse.ArgumentParser(description="Indic-Speak Sentence-Delimited Essay Streaming")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base URL of server")
    parser.add_argument("--text", default=None, help="Custom essay text (defaults to sample Hindi essay)")
    parser.add_argument("--speaker", default="Kavya", help="Voice name (default: Kavya)")
    parser.add_argument("--lang", default="hi-IN", help="Language code (default: hi-IN)")
    parser.add_argument("--out", default="essay_streamed.wav", help="Output concatenated WAV file")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    essay_text = args.text or (SAMPLE_ENGLISH_ESSAY if args.lang.startswith("en") else SAMPLE_HINDI_ESSAY)

    print("=" * 68)
    print("🚀 SENTENCE-DELIMITED ESSAY STREAMING PIPELINE")
    print("=" * 68)
    print(f"Target URL:    {base_url}")
    print(f"Speaker/Lang:  {args.speaker} ({args.lang})")

    sentences = split_sentences(essay_text)
    print(f"Total Sentences: {len(sentences)}\n")

    results = []
    full_pcm = bytearray()

    for idx, sent in enumerate(sentences, 1):
        preview = sent[:35] + ("..." if len(sent) > 35 else "")
        print(f"[*] Synthesizing Sentence {idx}/{len(sentences)}: '{preview}'")
        res = stream_sentence(base_url, sent, args.speaker, args.lang)
        results.append(res)
        full_pcm.extend(res["pcm"])
        print(f"    --> TTFA: {res['ttfa_ms']:.1f}ms | Audio: {res['audio_dur_sec']:.2f}s | Gen: {res['gen_sec']:.2f}s | RTF: {res['rtf']:.3f}")

    # Write complete concatenated WAV
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(full_pcm)

    total_audio = sum(r["audio_dur_sec"] for r in results)
    total_gen = sum(r["gen_sec"] for r in results)
    first_ttfa = results[0]["ttfa_ms"] if results else 0
    overall_rtf = total_gen / total_audio if total_audio > 0 else 0

    print("\n" + "=" * 68)
    print("📊 ESSAY STREAMING PIPELINE SUMMARY")
    print("=" * 68)
    print(f"{'Sent #':<8}{'Words':<8}{'TTFA (ms)':<12}{'Audio (s)':<12}{'Gen (s)':<10}{'RTF':<8}")
    print("-" * 68)
    for idx, r in enumerate(results, 1):
        words = len(r["text"].split())
        print(f"{idx:<8}{words:<8}{r['ttfa_ms']:<12.1f}{r['audio_dur_sec']:<12.2f}{r['gen_sec']:<10.2f}{r['rtf']:<8.3f}")
    print("-" * 68)
    print(f"First Sentence TTFA (Initial Voice Latency): {first_ttfa:.1f} ms ⚡")
    print(f"Total Essay Audio Duration:                  {total_audio:.2f} seconds")
    print(f"Total GPU Generation Time:                   {total_gen:.2f} seconds")
    print(f"Overall Direct RTF:                          {overall_rtf:.3f}")
    print(f"Saved Complete Essay Audio:                  {args.out}")
    print("=" * 68)


if __name__ == "__main__":
    main()
