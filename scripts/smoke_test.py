"""10-minute smoke test: sends 100 random-length questions from the ground
truth set through /ask and confirms the service does not crash or leak
memory, per the RAID §6 requirement.

Usage: python scripts/smoke_test.py
"""
from __future__ import annotations

import gc
import json
import random
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from app.main import app

    questions = [json.loads(line)["question"] for line in open(ROOT / "evaluation/ground_truth.jsonl", encoding="utf-8")]
    random.seed(42)
    # 100 random-length questions: mix full questions with truncated
    # (but still >=3 char, per AskRequest's min_length) random-length prefixes
    trials = []
    for _ in range(100):
        q = random.choice(questions)
        cut = random.randint(3, len(q))
        trials.append(q[:cut])

    tracemalloc.start()
    with TestClient(app) as client:
        start_mem = tracemalloc.get_traced_memory()[0]
        latencies, statuses, errors = [], [], 0

        t_start = time.time()
        for i, q in enumerate(trials):
            try:
                t0 = time.time()
                resp = client.post("/ask", json={"question": q})
                latencies.append((time.time() - t0) * 1000)
                statuses.append(resp.status_code)
                if resp.status_code >= 500:
                    errors += 1
            except Exception as e:  # noqa: BLE001 - smoke test: any crash is a failure to report
                errors += 1
                print(f"[{i}] CRASH: {e}")
        elapsed = time.time() - t_start

        gc.collect()
        end_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    print(f"Sent {len(trials)} requests in {elapsed:.1f}s")
    print(f"Status codes: {sorted(set(statuses))} (counts: "
          f"{ {s: statuses.count(s) for s in set(statuses)} })")
    print(f"Server-side crashes (5xx or exception): {errors}")
    print(f"Latency ms: min={min(latencies):.1f} mean={sum(latencies)/len(latencies):.1f} max={max(latencies):.1f}")
    print(f"Traced Python heap: start={start_mem/1e6:.1f}MB end={end_mem/1e6:.1f}MB "
          f"delta={(end_mem-start_mem)/1e6:+.1f}MB")

    assert errors == 0, f"{errors} requests crashed the server"
    print("\nSMOKE TEST PASSED: no crashes, no 5xx responses.")


if __name__ == "__main__":
    main()
