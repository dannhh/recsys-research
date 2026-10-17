"""Pilot: do next-item patterns in logs come from users or from the logging recommender? (Yambda-50M)

Yambda marks each listen as organic (user found the track) or recommendation-driven (served by the platform's
recommender). If sequential structure is largely produced by the recommender (autoplay, radio, curated streams),
then a pairwise transition model should predict recommendation-driven next listens much better than organic ones.

Setup: listens with played_ratio >= 50% ("listen+"); global temporal split, the last day is test.
Models from the training days: popularity, first-order transition counts (previous item -> next item), and a
recency-weighted pairwise sum over the last 5 items (PCTM-style, gamma_d = 1/sqrt(d)).
Metrics: HR@10 and HR@100 on sampled test events, split by the target's flag and by (previous flag, target flag).

Usage: python scripts/yambda_organic_pilot.py [--n_test 30000]
"""
import argparse

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from tiger import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=30000); ap.add_argument("--W", type=int, default=5)
    a = ap.parse_args()
    t = pq.read_table(ROOT / "data/raw/yambda/listens_50m.parquet",
                      columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"]).to_pandas()
    t = t[t.played_ratio_pct >= 50].sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)
    items, inv = np.unique(t.item_id.to_numpy(), return_inverse=True); t["item"] = inv.astype(np.int32)
    n = len(items)
    t["seen"] = t.duplicated(["uid", "item"])
    split = t.timestamp.max() - 86400
    test_mask = (t.timestamp > split).to_numpy()
    uid, item, org, seen = t.uid.to_numpy(), t.item.to_numpy(), t.is_organic.to_numpy(), t.seen.to_numpy()
    same_user_prev = np.r_[False, uid[1:] == uid[:-1]]
    print(f"listen+ events {len(t):,}; items {n:,}; organic share {org.mean():.3f}; test events {test_mask.sum():,}")

    # transition counts from training events only: pairs (i, i+d) of the same user, both in train
    tr = ~test_mask
    rows, cols, vals = [], [], []
    for d in range(1, a.W + 1):
        ok = np.zeros(len(t), dtype=bool); ok[d:] = (uid[d:] == uid[:-d]) & tr[d:] & tr[:-d]
        idx = np.nonzero(ok)[0]
        rows.append(item[idx - d]); cols.append(item[idx]); vals.append(np.full(len(idx), d ** -0.5, dtype=np.float32))
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    C.sum_duplicates()
    C1 = sp.csr_matrix((vals[0], (rows[0], cols[0])), shape=(n, n)); C1.sum_duplicates()
    pop = np.bincount(item[tr], minlength=n).astype(np.float64)
    pop_top = np.argsort(-pop)[:100]
    print(f"pairwise nnz {C.nnz:,}")

    # sample test events that have a previous event of the same user
    cand = np.nonzero(test_mask & same_user_prev)[0]
    rng = np.random.default_rng(0)
    ev = rng.choice(cand, min(a.n_test, len(cand)), replace=False)
    w = np.array([1 / np.sqrt(j + 1) for j in range(a.W)]); w /= w.sum()

    def rank_hits(score_row, target):
        r = score_row.tocoo() if sp.issparse(score_row) else score_row
        if r.nnz:
            order = r.col[np.argsort(-r.data)][:100]
            order = np.concatenate([order, pop_top[~np.isin(pop_top, order)]])[:100]
        else:
            order = pop_top
        pos = np.nonzero(order == target)[0]
        return (pos[0] if len(pos) else 10**9)

    res = {m: np.zeros(len(ev)) for m in ("pop", "markov1", "pairwise5")}
    for k, e in enumerate(ev):
        tgt = item[e]
        res["pop"][k] = (np.nonzero(pop_top == tgt)[0][0] if tgt in pop_top else 10**9)
        res["markov1"][k] = rank_hits(C1[item[e - 1]], tgt)
        hist = [j for j in range(e - 1, max(e - 1 - a.W, -1), -1) if uid[j] == uid[e]]
        row = None
        for jj, j in enumerate(hist):
            r = C[item[j]] * w[jj]
            row = r if row is None else row + r
        res["pairwise5"][k] = rank_hits(row, tgt)
    tgt_org, prev_org = org[ev], org[ev - 1]
    print(f"\nsampled test events {len(ev):,}; organic share among them {tgt_org.mean():.3f}")
    groups = {"all": np.ones(len(ev), bool), "target organic": tgt_org == 1, "target recommended": tgt_org == 0,
              "rec -> rec": (prev_org == 0) & (tgt_org == 0), "org -> org": (prev_org == 1) & (tgt_org == 1),
              "org -> rec": (prev_org == 1) & (tgt_org == 0), "rec -> org": (prev_org == 0) & (tgt_org == 1),
              "organic, repeat": (tgt_org == 1) & seen[ev], "recommended, repeat": (tgt_org == 0) & seen[ev],
              "organic, new": (tgt_org == 1) & ~seen[ev], "recommended, new": (tgt_org == 0) & ~seen[ev]}
    print(f"{'group':<20}{'n':>7} | " + " | ".join(f"{m:>9} HR@10  HR@100" for m in res))
    for g, msk in groups.items():
        print(f"{g:<20}{msk.sum():>7} | " + " | ".join(f"{(res[m][msk] < 10).mean():>15.4f} {(res[m][msk] < 100).mean():.4f}" for m in res))


if __name__ == "__main__":
    main()
