"""Amazon Reviews 2023, one category (default Musical_Instruments), official 5-core benchmark splits.

Builds, under data/processed/amazon2023_<category>/:
  items.json       item index -> {asin, text, category_path}
  loo.json         leave-one-out: {user: {"train": [...items...], "valid": item, "test": item}}
  ts.json          timestamp split: {"train"/"valid"/"test": [[history items...], target], ...}
  emb.npy          Sentence-T5 (768-d, normalised) text embeddings
  sids.npy         (n_items, 4): 3 Residual K-means codes + disambiguation code
  checks.json      data-sufficiency numbers (categories, unseen targets, group sizes)

SID codebooks are fitted only on items seen in the timestamp-split training period; later items
are assigned to the nearest centroids, so new-item experiments do not peek at the future.
Item text follows TIGER: title, brand (store), price, categories.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "amazon2023"
LEVELS, K, SEED = 3, 256, 0


def read(split, part, cat):
    return pd.read_csv(RAW / f"{cat}.{split}.{part}.csv", dtype={"history": str}).fillna({"history": ""})


def load_meta(asins, cat):
    meta = {}
    with open(RAW / f"meta_{cat}.jsonl") as f:
        for line in f:
            m = json.loads(line)
            if m["parent_asin"] in asins:
                meta[m["parent_asin"]] = m
    return meta


def item_text(m):
    cats = m.get("categories") or []
    parts = [("Title", m.get("title")), ("Brand", m.get("store")), ("Price", m.get("price")),
             ("Categories", ", ".join(cats))]
    return "; ".join(f"{k}: {v}" for k, v in parts if v), cats


def fit_residual_kmeans(x_fit, x_all):
    codes, r_fit, r_all = np.zeros((len(x_all), LEVELS), dtype=np.int64), x_fit.copy(), x_all.copy()
    for level in range(LEVELS):
        km = KMeans(n_clusters=K, n_init=1, random_state=SEED + level).fit(r_fit)
        lab_all = km.predict(r_all)
        codes[:, level] = lab_all
        r_fit = r_fit - km.cluster_centers_[km.labels_]
        r_all = r_all - km.cluster_centers_[lab_all]
        print(f"  level {level}: codes used by all items {len(np.unique(lab_all))}/{K}")
    seen, dis = defaultdict(int), np.zeros(len(x_all), dtype=np.int64)
    for i, t in enumerate(map(tuple, codes)):
        dis[i] = seen[t]; seen[t] += 1
    return np.concatenate([codes, dis[:, None]], axis=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--category", default="Musical_Instruments")
    cat = ap.parse_args().category
    out = ROOT / "data" / "processed" / f"amazon2023_{cat}"; out.mkdir(parents=True, exist_ok=True)

    lo = {p: read("last_out", p, cat) for p in ("train", "valid", "test")}
    ts = {p: read("timestamp_w_his", p, cat) for p in ("train", "valid", "test")}
    asins = set()
    for d in list(lo.values()) + list(ts.values()):
        asins |= set(d.parent_asin)
        if "history" in d:
            asins |= {a for h in d.history for a in h.split()}
    asins = sorted(asins); idx = {a: i for i, a in enumerate(asins)}
    meta = load_meta(set(asins), cat)
    texts, cats = zip(*(item_text(meta.get(a, {})) for a in asins))
    (out / "items.json").write_text(json.dumps([{"asin": a, "text": t, "category_path": c}
                                               for a, t, c in zip(asins, texts, cats)]))
    print(f"[{cat}] items {len(asins):,}; with metadata {len(meta):,}; example: {texts[0][:150]}")

    # leave-one-out sequences
    tr = lo["train"].sort_values(["user_id", "timestamp"], kind="stable")
    seq = tr.groupby("user_id").parent_asin.apply(lambda s: [idx[a] for a in s]).to_dict()
    va = dict(zip(lo["valid"].user_id, lo["valid"].parent_asin.map(idx)))
    te = dict(zip(lo["test"].user_id, lo["test"].parent_asin.map(idx)))
    loo = {u: {"train": seq.get(u, []), "valid": int(va[u]), "test": int(te[u])} for u in te}
    (out / "loo.json").write_text(json.dumps(loo))

    # timestamp split with history
    tsj = {p: [[[idx[a] for a in h.split()], int(idx[t])] for h, t in zip(d.history, d.parent_asin)]
           for p, d in ts.items()}
    (out / "ts.json").write_text(json.dumps(tsj))
    train_items = {t for _, t in tsj["train"]} | {i for h, _ in tsj["train"] for i in h}

    # embeddings and SIDs (codebooks fitted on timestamp-train items only)
    emb_path = out / "emb.npy"
    if not emb_path.exists():
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/sentence-t5-base", device="mps")
        np.save(emb_path, st.encode(list(texts), batch_size=128, show_progress_bar=False,
                                    normalize_embeddings=True).astype(np.float32))
    emb = np.load(emb_path)
    fit_idx = np.array(sorted(train_items))
    print(f"SID codebooks fitted on {len(fit_idx):,} timestamp-train items")
    sids = fit_residual_kmeans(emb[fit_idx], emb)
    np.save(out / "sids.npy", sids)

    # checks
    checks = {"items": len(asins), "items_with_meta": len(meta), "loo_users": len(loo),
              "loo_interactions": int(len(tr) + 2 * len(loo))}
    depth = pd.Series([len(c) for c in cats]).value_counts().sort_index().to_dict()
    checks["category_depth_hist"] = {int(k): int(v) for k, v in depth.items()}
    checks["category_nodes"] = {d: len({tuple(c[:d]) for c in cats if len(c) >= d}) for d in (1, 2, 3, 4)}
    checks["sid_nodes"] = {d: len({tuple(s[:d]) for s in sids}) for d in (1, 2, 3)}
    checks["sid_items_needing_disambiguation"] = int((sids[:, 3] > 0).sum())
    hist = np.array([len(v["train"]) for v in loo.values()])
    checks["loo_history_quantiles_10_25_50_75_90_99"] = np.quantile(hist, [.1, .25, .5, .75, .9, .99]).astype(int).tolist()
    checks["loo_unseen_test_share"] = float(np.mean([v["test"] not in set(v["train"]) and v["test"] not in train_items for v in loo.values()]))
    for p in ("valid", "test"):
        unseen = [t not in train_items for _, t in tsj[p]]
        checks[f"ts_{p}_targets"] = len(unseen)
        checks[f"ts_{p}_unseen"] = int(sum(unseen))
    for p in ("train", "valid", "test"):
        checks[f"ts_{p}_time_range"] = [str(pd.to_datetime(ts[p].timestamp.min(), unit="ms").date()),
                                        str(pd.to_datetime(ts[p].timestamp.max(), unit="ms").date())]
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
