"""Third dataset: policy-exposed vs randomly exposed clicks in KuaiRand-Pure (Kuaishou).

KuaiRand inserts uniformly random videos into users' feeds and flags them (is_rand = 1); all other exposures come
from the production policy. Here the comparison is policy-driven vs random exposure (not user-driven vs policy as in
Yambda / KuaiSAR). Events are clicked exposures from all three logs, merged per user by time.
Same count models and metrics as yambda_seq_pilot.py; "organic" in the output means POLICY-exposed (is_rand = 0)
and "recommended" means RANDOM (is_rand = 1). Test = last --days days, stratified sample.

Usage: python scripts/kuairand_pilot.py [--days 3] [--n_test 10000]
"""
import argparse
import json

import numpy as np
import pandas as pd

from tiger import ROOT
from yambda_seq_pilot import GROUPS, metrics, run_counts, run_neural, sample_events

D = ROOT / "data/raw/kuairand/KuaiRand-Pure/data"


def load():
    cols = ["user_id", "video_id", "time_ms", "is_click", "is_rand", "tab"]
    logs = [pd.read_csv(f, usecols=cols) for f in sorted(D.glob("log_*_pure.csv"))]
    t = pd.concat(logs, ignore_index=True)
    print(f"exposures {len(t):,}; random share {t.is_rand.mean():.3f}; click rate policy {t[t.is_rand == 0].is_click.mean():.3f}, "
          f"random {t[t.is_rand == 1].is_click.mean():.3f}")
    print("tab share by exposure type:\n", pd.crosstab(t.tab, t.is_rand, normalize="columns").round(3).T.to_string())
    t = t[t.is_click == 1]
    t = pd.DataFrame({"uid": t.user_id.to_numpy(), "item_id": t.video_id.to_numpy(), "timestamp": (t.time_ms // 1000).to_numpy(),
                      "is_organic": (1 - t.is_rand).to_numpy()})
    t = t.sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)
    items, inv = np.unique(t.item_id.to_numpy(), return_inverse=True)
    t["item"] = inv.astype(np.int32)
    t["seen"] = t.duplicated(["uid", "item"])
    return t, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3); ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--model", default="counts", choices=["counts", "sasrec", "gru4rec"]); ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    t, n = load()
    uid, org, seen = t.uid.to_numpy(), t.is_organic.to_numpy(), t.seen.to_numpy()
    print(f"clicks {len(t):,}; users {t.uid.nunique():,}; items {n:,}; policy share {org.mean():.3f}; "
          f"repeat rate policy {seen[org == 1].mean():.3f}, random {seen[org == 0].mean():.3f}")
    test_mask = (t.timestamp > t.timestamp.max() - a.days * 86400).to_numpy()
    ev = np.concatenate([sample_events(test_mask & (org == f), uid, a.n_test // 2, 0) for f in (1, 0)])
    print(f"test events {len(ev):,}")
    if a.model == "counts":
        res = {m: metrics(p, org[ev], seen[ev]) for m, p in run_counts(t, n, ev, test_days=a.days).items()}
    else:
        pos, best = run_neural(t, n, ev, a.model, a.epochs, test_days=a.days)
        res = {a.model: metrics(pos, org[ev], seen[ev])}
    (ROOT / "runs" / f"kuairand_pilot_{a.model}_{a.days}d.json").write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        print(f"\n{m}   (organic = POLICY exposure, recommended = RANDOM exposure)")
        for g in GROUPS:
            x = r[g]
            print(f"  {g:<20} n={x['n']:>5}  HR@10 {x['HR@10']:.4f}  NDCG@10 {x['NDCG@10']:.4f}  HR@100 {x['HR@100']:.4f}")


if __name__ == "__main__":
    main()
