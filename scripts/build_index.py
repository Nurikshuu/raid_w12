"""Idempotent, reproducible index build: raw HTML -> articles -> chunks ->
embeddings (disk-cached) -> FAISS index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --strategy recursive_char --chunk-size 800 --overlap 100
    python scripts/build_index.py --strategy article_based

Re-running with the same arguments and the same source files is a no-op cost
wise: the embedding cache is keyed by hash(text + model_name +
preprocessing_version), so unchanged chunks are never re-embedded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_lib as lib  # noqa: E402

PREPROCESSING_VERSION = "v1"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_chunks(strategy: str, chunk_size: int, overlap: int) -> list[lib.Chunk]:
    labor_articles = lib.parse_labor_code(ROOT / "data/raw/labor_code_raw.html")
    tax_articles = lib.parse_tax_code(ROOT / "data/raw/tax_code_raw.html")
    articles = labor_articles + tax_articles

    if strategy == "recursive_char":
        return lib.recursive_char_chunks(articles, chunk_size=chunk_size, chunk_overlap=overlap)
    elif strategy == "article_based":
        return lib.article_based_chunks(articles, max_chars=chunk_size)
    raise ValueError(f"unknown strategy: {strategy}")


def _chunk_hash(text: str, model_name: str) -> str:
    key = f"{text}|{model_name}|{PREPROCESSING_VERSION}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def embed_with_cache(chunks: list[lib.Chunk], model_name: str, cache_path: Path) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

    hashes = [_chunk_hash(c.text, model_name) for c in chunks]
    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]

    if missing_idx:
        print(f"Embedding cache: {len(chunks) - len(missing_idx)}/{len(chunks)} hits, "
              f"encoding {len(missing_idx)} new chunks...")
        model = SentenceTransformer(model_name)
        texts = [chunks[i].text for i in missing_idx]
        t0 = time.time()
        vecs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        print(f"Encoded {len(texts)} chunks in {time.time() - t0:.1f}s")
        for i, vec in zip(missing_idx, vecs):
            cache[hashes[i]] = vec.astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    else:
        print(f"Embedding cache: {len(chunks)}/{len(chunks)} hits, no encoding needed.")

    return np.stack([cache[h] for h in hashes]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["recursive_char", "article_based"], default="article_based")
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", default=str(ROOT / "data/index"))
    parser.add_argument("--chunks-path", default=str(ROOT / "data/chunks.jsonl"))
    parser.add_argument("--cache-path", default=str(ROOT / "data/cache/embedding_cache.pkl"))
    args = parser.parse_args()

    t_start = time.time()

    print(f"Parsing corpus + chunking (strategy={args.strategy}, chunk_size={args.chunk_size})...")
    chunks = build_chunks(args.strategy, args.chunk_size, args.overlap)
    print(f"Produced {len(chunks)} chunks.")

    chunks_path = Path(args.chunks_path)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    print(f"Persisted chunks to {chunks_path}")

    embeddings = embed_with_cache(chunks, args.model, Path(args.cache_path))

    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    meta = {
        "embedding_model": args.model,
        "strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "preprocessing_version": PREPROCESSING_VERSION,
        "num_chunks": len(chunks),
        "index_version": f"{args.strategy}-{args.chunk_size}-{args.model.split('/')[-1]}",
    }
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"Index built: {index.ntotal} vectors, dim={dim}. Total time: {elapsed:.1f}s")
    if elapsed > 900:
        print("WARNING: build exceeded the 15-minute CPU budget from the RAID spec.")


if __name__ == "__main__":
    main()
