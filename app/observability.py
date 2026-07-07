"""Structured JSON logging (stdout) + a rolling in-memory metrics window.

Every request logs one JSON line carrying request_id, prompt_version,
model_name, per-stage latency, token usage, cache_hit, and error status, per
the RAID observability spec. /metrics exposes p50/p95 latency, error rate,
cache hit rate, total tokens, and mean top-1 retrieval score over the last
100 requests.
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_request(logger: logging.Logger, **fields):
    logger.info("request_completed", extra={"extra_fields": fields})


@dataclass
class RequestRecord:
    total_latency_ms: float
    error: bool
    cache_hit: bool
    tokens: int
    top1_retrieval_score: float | None


@dataclass
class MetricsWindow:
    max_size: int = 100
    _records: deque = field(default_factory=lambda: deque(maxlen=100))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _total_requests: int = 0

    def record(self, rec: RequestRecord):
        with self._lock:
            self._records.append(rec)
            self._total_requests += 1

    def snapshot(self) -> dict:
        with self._lock:
            records = list(self._records)
            total_requests = self._total_requests

        if not records:
            return {
                "request_count": 0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "error_rate": 0.0,
                "cache_hit_rate": 0.0,
                "total_tokens": 0,
                "mean_top1_retrieval_score": 0.0,
            }

        latencies = sorted(r.total_latency_ms for r in records)

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = min(len(data) - 1, int(round(p * (len(data) - 1))))
            return data[idx]

        scores = [r.top1_retrieval_score for r in records if r.top1_retrieval_score is not None]
        return {
            "request_count": total_requests,
            "p50_latency_ms": round(percentile(latencies, 0.50), 2),
            "p95_latency_ms": round(percentile(latencies, 0.95), 2),
            "error_rate": round(sum(r.error for r in records) / len(records), 4),
            "cache_hit_rate": round(sum(r.cache_hit for r in records) / len(records), 4),
            "total_tokens": sum(r.tokens for r in records),
            "mean_top1_retrieval_score": round(statistics.mean(scores), 4) if scores else 0.0,
        }


metrics = MetricsWindow()
logger = configure_logging()
