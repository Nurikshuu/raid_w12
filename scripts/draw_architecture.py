"""One-off script that renders docs/architecture.png. Not part of the
runtime pipeline - run manually if the diagram needs regenerating."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7.5)
ax.axis("off")


def box(x, y, w, h, text, color="#4C72B0", fontsize=9.5, textcolor="white"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=color, edgecolor="#2c3e50", linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, wrap=True)
    return (x, y, w, h)


def arrow(b1, b2, side1="right", side2="left", label=None, color="#333"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    p1 = {"right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
          "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1)}[side1]
    p2 = {"right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
          "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2)}[side2]
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                  color=color, linewidth=1.3))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.15, label, fontsize=7.5, ha="center", color=color)


# offline pipeline (top row)
raw = box(0.3, 6.2, 1.8, 0.9, "adilet.zan.kz\nraw HTML\n(data/raw/)", color="#95A5A6")
ingest = box(2.5, 6.2, 2.0, 0.9, "ingest_lib.py\nparse + chunk", color="#95A5A6")
chunks = box(4.9, 6.2, 1.8, 0.9, "chunks.jsonl", color="#95A5A6")
build = box(7.0, 6.2, 2.2, 0.9, "build_index.py\nembed (cached) + FAISS", color="#95A5A6")
index = box(9.6, 6.2, 2.1, 0.9, "data/index/\nindex.faiss + meta.json", color="#95A5A6")

arrow(raw, ingest, "right", "left")
arrow(ingest, chunks, "right", "left")
arrow(chunks, build, "right", "left")
arrow(build, index, "right", "left")

# runtime path
user = box(0.3, 4.3, 1.6, 0.9, "User /\nCEO demo", color="#DD8452")
api = box(2.3, 4.3, 2.0, 0.9, "FastAPI\n/ask /health\n/ready /metrics", color="#4C72B0")
retr = box(4.7, 4.3, 2.0, 0.9, "Retriever\n(app/retrieval.py)\n+ TTL cache", color="#4C72B0")
gen = box(7.1, 4.3, 2.2, 0.9, "Generation\n(app/generation.py)\nAlem AI LLM", color="#4C72B0")

arrow(user, api, "right", "left", "POST /ask")
arrow(api, retr, "right", "left")
arrow(retr, gen, "right", "left", "top-k chunks")
arrow(gen, api, "bottom", "top", "JSON answer", color="#888")

# index feeds retriever at startup
arrow(index, retr, "bottom", "top", "load at startup\n(no re-embed)", color="#888")

# reliability + observability (bottom row)
rel = box(2.3, 2.5, 2.4, 0.9, "reliability.py\ntimeout + retry\n+ degradation", color="#C44E52")
obs = box(5.0, 2.5, 2.6, 0.9, "observability.py\nJSON logs (request_id)\nmetrics window", color="#55A868")
resp_cache = box(7.9, 2.5, 2.0, 0.9, "response cache\n(TTL, hash(question))", color="#8172B2")

arrow(rel, api, "top", "bottom", color="#C44E52")
arrow(api, obs, "bottom", "top", color="#55A868")
arrow(api, resp_cache, "bottom", "top", color="#8172B2")

ax.text(6, 1.2,
        "Offline (top): raw docs -> chunks -> cached embeddings -> FAISS index, run via `make index`.\n"
        "Runtime (middle): FastAPI loads the index once at startup and serves /ask with retrieval + response caching,\n"
        "bounded LLM timeout, retry-with-backoff, and graceful degradation if the index is unreachable.",
        ha="center", va="center", fontsize=9, color="#333")

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=150)
print("Saved docs/architecture.png")
