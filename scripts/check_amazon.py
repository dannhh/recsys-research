"""Data sufficiency checks for Amazon Reviews 2014 Beauty (5-core).

Answers, before any training:
  1. How many items have a human category path in the metadata, and how deep?
  2. How many unseen target items exist under leave-one-out vs a global temporal split?
  3. How large are the groups we want conditional coverage for (user activity, item popularity)?
"""
import ast
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_reviews():
    rows = []
    with gzip.open(RAW / "reviews_Beauty_5.json.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    return pd.DataFrame(rows, columns=["user", "item", "ts"])


def load_categories(items):
    cats = {}
    with gzip.open(RAW / "meta_Beauty.json.gz", "rt") as f:
        for line in f:
            m = ast.literal_eval(line)
            if m["asin"] in items:
                cats[m["asin"]] = m.get("categories", [])
    return cats


def main():
    df = load_reviews().sort_values(["user", "ts"], kind="stable")
    items = set(df["item"])
    print(f"users {df.user.nunique():,}  items {len(items):,}  interactions {len(df):,}")
    print(f"time span {pd.to_datetime(df.ts.min(), unit='s').date()} .. {pd.to_datetime(df.ts.max(), unit='s').date()}")

    # 1. Human category coverage
    cats = load_categories(items)
    paths = {a: max(c, key=len) if c else [] for a, c in cats.items()}  # longest path per item
    depth = Counter(len(p) for p in paths.values())
    with_cat = sum(1 for p in paths.values() if len(p) >= 2)
    print(f"\n[categories] items with metadata {len(cats):,}/{len(items):,}; "
          f"with a path of depth >= 2: {with_cat:,} ({with_cat / len(items):.1%})")
    print("  depth distribution:", dict(sorted(depth.items())))
    for d in (2, 3, 4):
        nodes = {tuple(p[:d]) for p in paths.values() if len(p) >= d}
        print(f"  distinct nodes at depth {d}: {len(nodes):,}")

    # 2a. Leave-one-out: last item of each user is the test target
    last = df.groupby("user").tail(1)
    train_loo = df.drop(last.index)
    seen = set(train_loo["item"])
    unseen_loo = (~last["item"].isin(seen)).mean()
    print(f"\n[leave-one-out] test targets {len(last):,}; unseen-item share {unseen_loo:.2%}")

    # 2b. Global temporal split: test = interactions after the 90% time quantile
    t_cut = df["ts"].quantile(0.90)
    train_t, test_t = df[df.ts <= t_cut], df[df.ts > t_cut]
    seen_t = set(train_t["item"])
    test_users = test_t.groupby("user").head(1)  # first test interaction per user
    test_users = test_users[test_users.user.isin(set(train_t.user))]  # need a history
    unseen_t = ~test_users["item"].isin(seen_t)
    print(f"[global temporal, cut at {pd.to_datetime(t_cut, unit='s').date()}] "
          f"test targets with history {len(test_users):,}; seen {int((~unseen_t).sum()):,}; "
          f"unseen {int(unseen_t.sum()):,} ({unseen_t.mean():.1%})")

    # 3. Group sizes for conditional coverage (leave-one-out test users)
    hist_len = train_loo.groupby("user").size().reindex(last.user).to_numpy()
    q = np.quantile(hist_len, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    print(f"\n[user activity] history length quantiles 10/25/50/75/90/99%: {q.astype(int).tolist()}")
    for lo, hi in [(4, 5), (6, 10), (11, 20), (21, 10**9)]:
        n = int(((hist_len >= lo) & (hist_len <= hi)).sum())
        print(f"  history {lo}-{hi if hi < 10**9 else 'max'}: {n:,} test users")
    pop = train_loo["item"].value_counts()
    tgt_pop = last["item"].map(pop).fillna(0).to_numpy()
    for name, lo, hi in [("tail (<=5)", 0, 5), ("mid (6-20)", 6, 20), ("head (>20)", 21, 10**9)]:
        n = int(((tgt_pop >= lo) & (tgt_pop <= hi)).sum())
        print(f"  target popularity {name}: {n:,} test users")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "amazon_beauty_checks.json").write_text(json.dumps({
        "users": int(df.user.nunique()), "items": len(items), "interactions": len(df),
        "category_depth_ge2_share": with_cat / len(items),
        "loo_unseen_share": float(unseen_loo),
        "temporal_test_targets": int(len(test_users)), "temporal_unseen_share": float(unseen_t.mean()),
    }, indent=1))


if __name__ == "__main__":
    main()
