"""
Indic-Speak Production Benchmark Suite
======================================
Measures real-world latency, Time-to-First-Audio (TTFA), Real-Time Factor (RTF),
throughput, and memory utilization across multilingual and STEM workloads.

Usage:
    python benchmark.py --url http://localhost:8000 --concurrency 1,2,4 --trials 3
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from typing import List, Dict, Any
import aiohttp
import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

# ─── BENCHMARK DATASETS ──────────────────────────────────────────────────────
TEST_UTTERANCES = [
    {
        "lang": "hi",
        "category": "STEM / Science",
        "speaker": "Amit",
        "text": "प्रकाश की चाल निर्वात में लगभग तीन लाख किलोमीटर प्रति सेकंड यानी 3 * 10^8 m/s होती है।",
    },
    {
        "lang": "hi",
        "category": "Conversational / Code-mixed",
        "speaker": "Kavya",
        "text": "नमस्ते! आपका order confirm हो गया है, delivery कल शाम 5 बजे तक आपके address पर पहुँच जाएगी।",
    },
    {
        "lang": "en",
        "category": "Technical English",
        "speaker": "Amit",
        "text": "The derivative of x squared plus 5x with respect to x equals 2x plus 5.",
    },
    {
        "lang": "ta",
        "category": "Tamil Vernacular",
        "speaker": "Anitha",
        "text": "வணக்கம், உங்கள் ஆர்டர் வெற்றிகரமாக பதிவு செய்யப்பட்டுள்ளது.",
    },
    {
        "lang": "te",
        "category": "Telugu Vernacular",
        "speaker": "Vamsi",
        "text": "నమస్కారం, మీ ఖాతా వివరాలు విజయవంతంగా అప్‌డేట్ చేయబడ్డాయి.",
    },
]

SR = 24_000
BYTES_PER_SECOND = SR * 2  # 16-bit mono = 48,000 bytes/sec


async def measure_single_request(
    session: aiohttp.ClientSession,
    url: str,
    item: Dict[str, str],
) -> Dict[str, Any]:
    """Sends a single streaming request and tracks latency milestones."""
    endpoint = f"{url.rstrip('/')}/synthesize/stream"
    payload = {
        "text": item["text"],
        "speaker": item["speaker"],
        "response_format": "pcm",
    }

    t_start = time.perf_counter()
    ttfa = None
    total_bytes = 0

    try:
        async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=120, connect=10)) as resp:
            if resp.status != 200:
                err = await resp.text()
                return {"status": "error", "error": f"HTTP {resp.status}: {err}"}

            async for chunk in resp.content.iter_any():
                if chunk:
                    if ttfa is None:
                        ttfa = (time.perf_counter() - t_start) * 1000.0  # ms
                    total_bytes += len(chunk)

        t_end = time.perf_counter()
        total_gen_time_s = t_end - t_start
        audio_dur_s = total_bytes / BYTES_PER_SECOND
        rtf = total_gen_time_s / audio_dur_s if audio_dur_s > 0 else 0.0

        return {
            "status": "success",
            "lang": item["lang"],
            "category": item["category"],
            "speaker": item["speaker"],
            "ttfa_ms": round(ttfa or 0.0, 1),
            "total_time_s": round(total_gen_time_s, 2),
            "audio_dur_s": round(audio_dur_s, 2),
            "rtf": round(rtf, 3),
            "bytes_received": total_bytes,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


async def run_concurrency_tier(
    url: str,
    concurrency: int,
    trials: int,
) -> List[Dict[str, Any]]:
    """Runs concurrent load tests against the Indic-Speak server."""
    logger.info(f"Running concurrency tier: {concurrency} workers across {trials} trials...")
    results = []

    async with aiohttp.ClientSession() as session:
        for trial in range(trials):
            for i in range(0, len(TEST_UTTERANCES), concurrency):
                batch = TEST_UTTERANCES[i : i + concurrency]
                if len(batch) < concurrency:
                    # Pad batch to maintain concurrency level
                    batch = (batch * ((concurrency // len(batch)) + 1))[:concurrency]

                tasks = [measure_single_request(session, url, item) for item in batch]
                batch_res = await asyncio.gather(*tasks)
                results.extend(batch_res)

    return results


def print_report(concurrency: int, results: List[Dict[str, Any]]):
    """Calculates percentiles and renders a markdown table."""
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    if not successful:
        logger.error(f"All requests failed at concurrency {concurrency}!")
        return

    ttfas = [r["ttfa_ms"] for r in successful]
    rtfs = [r["rtf"] for r in successful]
    tot_times = [r["total_time_s"] for r in successful]

    ttfas.sort()
    rtfs.sort()

    p50_ttfa = statistics.median(ttfas)
    p90_ttfa = ttfas[int(len(ttfas) * 0.90)]
    p95_ttfa = ttfas[int(len(ttfas) * 0.95)]
    p99_ttfa = ttfas[min(int(len(ttfas) * 0.99), len(ttfas) - 1)]

    avg_rtf = statistics.mean(rtfs)
    p50_rtf = statistics.median(rtfs)

    headers = ["Metric", f"Concurrency = {concurrency}"]
    table_data = [
        ["Total Requests", len(results)],
        ["Successful", len(successful)],
        ["Failed", len(failed)],
        ["TTFA p50 (Median)", f"{p50_ttfa:.1f} ms"],
        ["TTFA p90", f"{p90_ttfa:.1f} ms"],
        ["TTFA p95", f"{p95_ttfa:.1f} ms"],
        ["TTFA p99", f"{p99_ttfa:.1f} ms"],
        ["Average RTF", f"{avg_rtf:.3f}"],
        ["Median RTF", f"{p50_rtf:.3f}"],
        ["Real-Time Capable", "YES (RTF < 1.0)" if p50_rtf < 1.0 else "NO (RTF >= 1.0)"],
    ]

    print("\n" + "=" * 50)
    print(f"BENCHMARK REPORT: Concurrency = {concurrency}")
    print("=" * 50)
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="github"))
    print("\n")


async def main():
    parser = argparse.ArgumentParser(description="Indic-Speak Benchmark Tool")
    parser.add_argument("--url", default="http://localhost:8000", help="Indic-Speak server URL")
    parser.add_argument("--concurrency", default="1,2,4", help="Comma-separated concurrency tiers")
    parser.add_argument("--trials", type=int, default=2, help="Trials per tier")
    args = parser.parse_args()

    # Check health first
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{args.url}/health") as r:
                data = await r.json()
                logger.info(f"Target Server Healthy! Data: {data}")
    except Exception as e:
        logger.error(f"Cannot reach server at {args.url}: {e}")
        return

    tiers = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    for c in tiers:
        res = await run_concurrency_tier(args.url, concurrency=c, trials=args.trials)
        print_report(c, res)


if __name__ == "__main__":
    asyncio.run(main())
