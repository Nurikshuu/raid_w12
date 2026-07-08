# ADR 001: Chunking strategy for the Kazakh Labor + Tax Code corpus

## Context

The corpus is two Kazakh statutes (Labor Code, Tax Code) fetched from `adilet.zan.kz`.
Both are organized as numbered, self-contained articles (`Статья N. Title ...`),
each grouped under a chapter (`Глава N. ...`). We need a chunking strategy that
maximizes `recall@5` on our 17-question ground truth set while keeping the
index build under the 15-minute CPU budget.

The RAID spec requires at least two chunking strategies be implemented and
compared, and (since this corpus is the "structured legal text" option) at
least two ablations if baseline recall clears 0.85.

## Options considered

1. **`RecursiveCharacterTextSplitter` at a fixed size** (200/400/800/1600 chars,
   ~10-15% overlap), applied per-article so a chunk never straddles two
   articles. Simple, required by the spec, no domain knowledge needed.
2. **`article_based`**: one chunk per article; articles longer than 1600 chars
   are split on their own numbered sub-clauses (`1.`, `2.`, `1-1.`, ...) rather
   than a blind character cut, respecting the structure the law itself already
   provides.
3. **Semantic chunking** (embedding-similarity-based splitting): rejected up
   front. It adds an extra embedding pass just to decide chunk boundaries, and
   buys nothing over `article_based` on a corpus where the boundary the model
   would rediscover (the article) is already given to us for free in the
   markup.

## Decision

We ship **`article_based`** (max 1600 chars per chunk) as the production
strategy used by `scripts/build_index.py`, and keep `recursive_char` as the
required second strategy for comparison in `notebooks/01_ingestion.ipynb` /
`notebooks/02_ablation.ipynb`.

Ablation 1 (`notebooks/02_ablation.ipynb`, chunk size/strategy sweep, embedding
model fixed at `paraphrase-multilingual-MiniLM-L12-v2`) measured recall@5 for
five variants; see `evaluation/results/ablation_chunk_size.png` and
`ablation_summary.json` for exact numbers. Recall climbs monotonically with
chunk size (0.647 at 200 chars -> 0.882 at 1600 chars) and plateaus once
chunks are large enough to hold a full article - splitting an article into
200-800 character windows sometimes separates a number (a rate, a threshold,
a duration) from the clause that explains what it means, which costs recall
for no benefit, since we never needed a smaller unit than "one article" in
the first place.

**Honest caveat:** at 1600 characters, `recursive_char_1600` (0.882 recall,
0.690 MRR) and `article_based` (0.882 recall, 0.678 MRR) are statistically
tied - chunk *size*, not the "respect article boundaries" heuristic, is
doing most of the work once chunks are big enough. `article_based` wins on
criteria the recall metric doesn't capture: 18% fewer chunks (1298 vs 1577),
~14% faster to build, and an exact chunk-to-article mapping by construction
rather than as a byproduct of a large-enough character window.

## Consequences

- **Pro**: highest recall@5 of any variant tested (tied with `recursive_char_1600`),
  zero extra chunking parameters to tune per corpus, and chunk metadata
  (`article_number`, `section_title`) is exact - useful for citations in the
  API response.
- **Con**: this strategy is corpus-specific. It assumes the source documents
  are already organized into short, numbered, self-contained units. It would
  need to fall back to `recursive_char` (or a smarter section-based splitter)
  for a corpus without that structure, e.g. free-form PDFs or prose reports
  (the SEC 10-K option in this RAID would not benefit from `article_based` at
  all).
- **Con**: a handful of tax articles run past 1600 chars (the longest is ~48k
  chars - long lists of exemptions); the sub-clause split for those is a
  second, weaker heuristic that hasn't been separately ablated.
- Chunk size 1600 for the sub-clause fallback was not itself tuned; it was
  chosen to match the `recursive_char_1600` variant in the sweep for a fair
  side-by-side comparison. Retuning it is the natural next ablation if a
  future corpus revision adds longer, less clause-structured articles.
