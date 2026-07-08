"""Measures the response cache's effect on p95 latency: sends the same 50
questions through /ask twice (first pass = cold / cache miss on every
question, second pass = warm / cache hit on every question) and reports
p50/p95 for each pass. Numbers go straight into the README "Caching" section.

Usage: python scripts/bench_cache.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_QUESTIONS = [json.loads(line)["question"] for line in open(ROOT / "evaluation/ground_truth.jsonl", encoding="utf-8")]

PREFIXES = ["", "Подскажите: ", "Согласно кодексу, ", "Уточните, пожалуйста: "]


def build_50_questions() -> list[str]:
    out = []
    i = 0
    while len(out) < 50:
        q = BASE_QUESTIONS[i % len(BASE_QUESTIONS)]
        prefix = PREFIXES[(i // len(BASE_QUESTIONS)) % len(PREFIXES)]
        out.append(prefix + q)
        i += 1
    return out


def percentile(data: list[float], p: float) -> float:
    data = sorted(data)
    idx = min(len(data) - 1, int(round(p * (len(data) - 1))))
    return data[idx]


def main():
    from fastapi.testclient import TestClient
    from app.main import app

    questions = build_50_questions()

    with TestClient(app) as client:
        def run_pass(label: str) -> list[float]:
            latencies = []
            cache_hits = 0
            for q in questions:
                t0 = time.time()
                resp = client.post("/ask", json={"question": q})
                latencies.append((time.time() - t0) * 1000)
                if resp.status_code == 200 and resp.json().get("cache_hit"):
                    cache_hits += 1
            print(f"[{label}] n={len(latencies)} cache_hits={cache_hits} "
                  f"p50={percentile(latencies, 0.50):.1f}ms p95={percentile(latencies, 0.95):.1f}ms")
            return latencies

        cold = run_pass("pass 1 - cold (cache empty)")
        warm = run_pass("pass 2 - warm (same 50 questions repeated)")

    result = {
        "cold": {"p50_ms": percentile(cold, 0.50), "p95_ms": percentile(cold, 0.95)},
        "warm": {"p50_ms": percentile(warm, 0.50), "p95_ms": percentile(warm, 0.95)},
    }
    out_path = ROOT / "evaluation/results/cache_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
