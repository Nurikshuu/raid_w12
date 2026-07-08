# Defense quick-reference

Notes allowed during the defense (per the RAID rules: no LLM, but your own
notes/code/ADRs/README are fine). Numbers here are pulled from
`evaluation/results/` — if asked to reproduce, re-run the referenced script.

## The three CEO questions

1. **How do you know it works?** `recall@5 = 0.882`, `MRR = 0.678` on 17
   hand-authored questions (`evaluation/evaluate_retrieval.py`). Prompt
   regression suite: 7/7 passing against the real LLM backend
   (`tests/test_prompt.py`). 100-question smoke test: 0 crashes, 0 5xx,
   +15.9MB heap growth (`scripts/smoke_test.py`).
2. **What happens when it breaks?** README §8 — three failure modes, each
   with an exact HTTP status/body and a unit test. Most likely first:
   LLM timeout/transient failure (third-party dependency), least likely:
   Pydantic validation (fully local, deterministic).
3. **How much does it cost?** ~2,300 tokens/request, cold p95 = 4,093ms
   (LLM-bound), warm (cache hit) p95 = 2.6ms. Reference cost estimate
   ≈ $0.46/1,000 questions at a representative small-model rate (Alem AI
   itself is a free educational endpoint with no published price).

## Numbers to have cold

| Question | Answer |
|---|---|
| recall@5 | 0.882 |
| MRR | 0.678 |
| Chunk size sweep range | 200/400/800/1600 chars, `recursive_char` + `article_based` |
| Best chunking strategy | `article_based` (tied with `recursive_char_1600` on recall, wins on chunk count/build speed/exact citations) |
| Embedding model comparison | multilingual 0.882 recall vs. English-only 0.118 recall, ~same encode time |
| Corpus size | 221 Labor Code articles + 405 curated Tax Code articles = 626 articles, 1,298 chunks |
| Index build time | ~75s (well under the 15-min budget) |
| Startup to `/ready` | ~17s (real uvicorn, not TestClient) - under 30s budget |
| Cache effect | p95 4,093ms (cold) -> 2.6ms (warm), TTL invalidation (1h default) |
| Reliability pattern | timeout (20s) + retry (3x, 1s/2s/4s backoff) + graceful degradation on retrieval failure |
| Prompt version | `rag_v1`, logged with every request |

## Likely "modify evaluate_retrieval.py live" asks

- Add a new metric (e.g. `precision@5` or `nDCG@10`) — the per-question loop
  in `evaluate()` already has `ranked_chunks` available; a new metric is a
  few lines inside that loop, same pattern as the existing `reciprocal_ranks`
  accumulator.
- Change `k` — it's already a CLI arg (`--k`) and a function parameter.
- Add a new ground-truth question — append a line to
  `evaluation/ground_truth.jsonl` with `id`, `category`, `question`,
  `relevant: [{law, article_number}]`; no code change needed.

## Things I might get asked to defend and haven't over-rehearsed

- Why `IndexFlatIP` and not HNSW: corpus is small (1,298 vectors), exact
  search is fast enough (sub-30ms) and simpler to reason about; HNSW is the
  first thing to swap in at 1M-document scale (see README "Scaling").
- Why TTL over index-version-bump for cache invalidation: the corpus (a
  codified law) changes on the order of months, not minutes - TTL is the
  simplest strategy that matches how often correctness actually matters here.
- Why the tax code is curated to ~5 chapters instead of the full ~830
  articles: keeps the corpus focused on what a fintech's employees would
  actually ask, keeps index build inside the 15-min CPU budget, and keeps the
  ground-truth set honestly labelable by one person in an afternoon.
