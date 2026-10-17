"""Amazon Beauty 2014 (5-core): user sequences, item texts, text embeddings, Semantic IDs.

Follows TIGER (Rajput et al., NeurIPS 2023): item text = title, price, brand, categories;
Sentence-T5 embeddings (768-d); 3 residual levels x 256 codes + 1 disambiguation code.
TIGER uses an RQ-VAE; we use Residual K-means (what NEO / GLIDE use; RQ-VAE is an ablation later).

Outputs (data/processed/beauty/):
  sequences.json  user -> chronological item index list
  items.json      item index -> {asin, text, category_path}
  emb.npy         (n_items, 768) float32
  sids.npy        (n_items, 4) int: 3 semantic codes + disambiguation code
"""
import ast
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed" / "beauty"
LEVELS, K, SEED = 3, 256, 0


def load():
    events = []
    with gzip.open(RAW / "reviews_Beauty_5.json.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            events.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    asins = sorted({a for _, a, _ in events})
    idx = {a: i for i, a in enumerate(asins)}
    meta = {}
    with gzip.open(RAW / "meta_Beauty.json.gz", "rt") as f:
        for line in f:
            m = ast.literal_eval(line)
            if m["asin"] in idx:
                meta[m["asin"]] = m
    seqs = defaultdict(list)
    for u, a, t in sorted(events, key=lambda e: (e[0], e[2])):  # stable within equal timestamps
        seqs[u].append(idx[a])
    return asins, meta, dict(seqs)


def item_text(m):
    cats = max(m.get("categories", [[]]), key=len)
    parts = [f"Title: {m.get('title', '')}", f"Brand: {m.get('brand', '')}",
             f"Price: {m.get('price', '')}", f"Categories: {', '.join(cats)}"]
    return "; ".join(p for p in parts if not p.endswith(": ")), cats


def residual_kmeans(x):
    codes, r = np.zeros((len(x), LEVELS), dtype=np.int64), x.copy()
    for level in range(LEVELS):
        km = KMeans(n_clusters=K, n_init=1, random_state=SEED + level).fit(r)
        codes[:, level] = km.labels_
        r = r - km.cluster_centers_[km.labels_]
        print(f"  level {level}: used codes {len(np.unique(km.labels_))}/{K}, residual norm {np.linalg.norm(r, axis=1).mean():.4f}")
    # disambiguation code: rank within each identical 3-code tuple (TIGER's 4th token)
    seen, dis = defaultdict(int), np.zeros(len(x), dtype=np.int64)
    for i, t in enumerate(map(tuple, codes)):
        dis[i] = seen[t]; seen[t] += 1
    return np.concatenate([codes, dis[:, None]], axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    asins, meta, seqs = load()
    texts, cats = zip(*(item_text(meta[a]) for a in asins))
    print(f"users {len(seqs):,} items {len(asins):,}; example text: {texts[0][:160]}")
    (OUT / "sequences.json").write_text(json.dumps(seqs))
    (OUT / "items.json").write_text(json.dumps([{"asin": a, "text": t, "category_path": c}
                                                for a, t, c in zip(asins, texts, cats)]))
    emb_path = OUT / "emb.npy"
    if not emb_path.exists():
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/sentence-t5-base", device="mps")
        emb = st.encode(list(texts), batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        np.save(emb_path, emb.astype(np.float32))
    emb = np.load(emb_path)
    sids = residual_kmeans(emb)
    np.save(OUT / "sids.npy", sids)
    tuples3 = {tuple(s[:3]) for s in sids}
    print(f"SIDs: {len(tuples3):,} distinct 3-code prefixes for {len(sids):,} items; "
          f"items needing disambiguation {int((sids[:, 3] > 0).sum()):,}; max 4th code {sids[:, 3].max()}")
    for d in (1, 2, 3):
        n = len({tuple(s[:d]) for s in sids})
        print(f"  depth {d}: {n:,} nodes, {len(sids) / n:.1f} items per node")


if __name__ == "__main__":
    main()
