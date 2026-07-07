"""Measures recall@5 and MRR of the retrieval pipeline against
evaluation/ground_truth.jsonl.

Usage:
    python evaluation/evaluate_retrieval.py
    python evaluation/evaluate_retrieval.py --k 5 --index-dir data/index --out evaluation/results/latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_ground_truth(gt_path: Path) -> list[dict]:
    items = []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def is_relevant(chunk: dict, relevant_set: set[tuple[str, str]]) -> bool:
    return (chunk["law"], chunk["article_number"]) in relevant_set


def evaluate(index_dir: Path, chunks_path: Path, gt_path: Path, k: int = 5, mrr_cutoff: int = 10) -> dict:
    import faiss
    from sentence_transformers import SentenceTransformer

    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(meta["embedding_model"])
    index = faiss.read_index(str(index_dir / "index.faiss"))
    chunks = load_chunks(chunks_path)
    gt = load_ground_truth(gt_path)

    per_question = []
    hits_at_k = 0
    reciprocal_ranks = []
    latencies_ms = []

    for item in gt:
        relevant_set = {(r["law"], r["article_number"]) for r in item["relevant"]}
        t0 = time.time()
        qvec = model.encode([item["question"]], normalize_embeddings=True).astype(np.float32)
        scores, idxs = index.search(qvec, max(k, mrr_cutoff))
        latencies_ms.append((time.time() - t0) * 1000)

        ranked_chunks = [chunks[i] for i in idxs[0]]
        top_k_hit = any(is_relevant(c, relevant_set) for c in ranked_chunks[:k])
        hits_at_k += int(top_k_hit)

        rr = 0.0
        for rank, c in enumerate(ranked_chunks[:mrr_cutoff], start=1):
            if is_relevant(c, relevant_set):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "hit_at_k": top_k_hit,
            "reciprocal_rank": rr,
            "top_chunk_ids": [c["chunk_id"] for c in ranked_chunks[:k]],
        })

    n = len(gt)
    result = {
        "index_meta": meta,
        "k": k,
        "num_questions": n,
        "recall_at_k": hits_at_k / n,
        "mrr": sum(reciprocal_ranks) / n,
        "mean_query_latency_ms": sum(latencies_ms) / n,
        "per_question": per_question,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(ROOT / "data/index"))
    parser.add_argument("--chunks-path", default=str(ROOT / "data/chunks.jsonl"))
    parser.add_argument("--ground-truth", default=str(ROOT / "evaluation/ground_truth.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "evaluation/results/latest.json"))
    args = parser.parse_args()

    result = evaluate(Path(args.index_dir), Path(args.chunks_path), Path(args.ground_truth), k=args.k)

    print(f"recall@{args.k} = {result['recall_at_k']:.3f}")
    print(f"MRR          = {result['mrr']:.3f}")
    print(f"mean query latency = {result['mean_query_latency_ms']:.1f} ms")
    misses = [q for q in result["per_question"] if not q["hit_at_k"]]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for m in misses:
            print(f"  [{m['id']}] {m['question']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
