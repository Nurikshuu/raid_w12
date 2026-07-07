"""Retrieval quality gate: recall@5 on the 17-question ground truth set must
be >= 0.75 to pass the RAID defense. Requires `make index` to have been run
first (data/index/index.faiss + data/chunks.jsonl must exist)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from evaluate_retrieval import evaluate  # noqa: E402

INDEX_DIR = ROOT / "data/index"
CHUNKS_PATH = ROOT / "data/chunks.jsonl"
GROUND_TRUTH = ROOT / "evaluation/ground_truth.jsonl"


@pytest.mark.skipif(
    not (INDEX_DIR / "index.faiss").exists(),
    reason="index not built yet - run `make index` first",
)
def test_recall_at_5_meets_defense_threshold():
    result = evaluate(INDEX_DIR, CHUNKS_PATH, GROUND_TRUTH, k=5)
    assert result["recall_at_k"] >= 0.75, (
        f"recall@5 = {result['recall_at_k']:.3f} is below the required 0.75 threshold"
    )


@pytest.mark.skipif(
    not (INDEX_DIR / "index.faiss").exists(),
    reason="index not built yet - run `make index` first",
)
def test_mrr_is_reported_and_positive():
    result = evaluate(INDEX_DIR, CHUNKS_PATH, GROUND_TRUTH, k=5)
    assert result["mrr"] > 0.0
