"""Loads the FAISS index + chunk store built by scripts/build_index.py and
serves top-k retrieval with an in-memory retrieval cache.

Retrieval cache key: hash(query + index_version + top_k) - a hit skips both
the query embedding call and the FAISS search.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    law: str
    article_number: str
    section_title: str
    text: str
    score: float


class TTLCache:
    """Small LRU + TTL cache. Good enough for a single-process API; documented
    invalidation strategy is TTL expiry (see README "Caching" section)."""

    def __init__(self, max_size: int, ttl_seconds: float | None = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            ts, value = entry
            if self.ttl_seconds is not None and (time.time() - ts) > self.ttl_seconds:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: object):
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class Retriever:
    def __init__(self, index_dir: Path, chunks_path: Path, model_name: str, cache_size: int):
        self.index_dir = index_dir
        self.chunks_path = chunks_path
        self.model_name = model_name
        self._index = None
        self._model = None
        self._chunks: list[dict] = []
        self.index_version = "unloaded"
        self.cache = TTLCache(max_size=cache_size, ttl_seconds=None)

    def load(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.index_version = meta["index_version"]

        self._index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self._model = SentenceTransformer(self.model_name)
        with open(self.chunks_path, encoding="utf-8") as f:
            self._chunks = [json.loads(line) for line in f]

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and self._model is not None and bool(self._chunks)

    def _cache_key(self, question: str, top_k: int) -> str:
        raw = f"{question}|{self.index_version}|{top_k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], bool, dict]:
        """Returns (chunks, cache_hit, timing_ms)."""
        if not self.is_loaded:
            raise RuntimeError("retriever not loaded: call .load() first")

        k = top_k or settings.top_k
        key = self._cache_key(question, k)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True, {"embedding_ms": 0.0, "search_ms": 0.0}

        import numpy as np

        t0 = time.time()
        qvec = self._model.encode([question], normalize_embeddings=True).astype(np.float32)
        embedding_ms = (time.time() - t0) * 1000

        t1 = time.time()
        scores, idxs = self._index.search(qvec, k)
        search_ms = (time.time() - t1) * 1000

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            c = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    law=c["law"],
                    article_number=c["article_number"],
                    section_title=c["section_title"],
                    text=c["text"],
                    score=float(score),
                )
            )
        self.cache.set(key, results)
        return results, False, {"embedding_ms": embedding_ms, "search_ms": search_ms}


retriever = Retriever(
    index_dir=settings.index_dir,
    chunks_path=settings.chunks_path,
    model_name=settings.embedding_model,
    cache_size=settings.retrieval_cache_size,
)
