"""Second dataset: user-driven (search) vs policy-driven (recommendation feed) events in KuaiSAR (Kuaishou).

KuaiSAR logs every action as coming from the search service (S) or the recommendation feed (R). The analogue of
Yambda's organic flag is the search click: the user typed or picked a query and chose a result, versus a video the
feed served. Same protocol and code as yambda_seq_pilot.py: events of both sources merged per user by time, the last
day is test, count models (popularity, first-order Markov, recency-weighted pairwise over the last 5) fitted on
earlier days, HR@10 / NDCG@10 / HR@100 per group (source x repeat/new). "organic" in the output means search.

Events kept: feed videos with click = 1 (played) and search results with click_cnt > 0.
Usage: python scripts/kuaisar_pilot.py [--n_test 10000]
"""
import argparse
import json

import numpy as np
import pandas as pd

from tiger import ROOT
from yambda_seq_pilot import GROUPS, metrics, run_counts, sample_events

D = ROOT / "data/raw/kuaisar/KuaiSAR_v2"


def load():
    r = pd.read_csv(D / "rec_inter.csv", usecols=["user_id", "item_id", "timestamp", "click"])
    r = r[r.click == 1]
    r = pd.DataFrame({"uid": r.user_id, "item_id": r.item_id, "ts": (r.timestamp // 1000).astype(np.int64), "is_organic": 0})
    s = pd.read_csv(D / "src_inter.csv", usecols=["user_id", "item_id", "click_cnt", "time"])
    s = s[s.click_cnt > 0]
    s = pd.DataFrame({"uid": s.user_id, "item_id": s.item_id,
                      "ts": (pd.to_datetime(s.time).astype("int64") // 10**9 - 8 * 3600).astype(np.int64),  # Beijing time -> UTC
                      "is_organic": 1})
    t = pd.concat([r, s], ignore_index=True).sort_values(["uid", "ts"], kind="stable").reset_index(drop=True)
    items, inv = np.unique(t.item_id.to_numpy(), return_inverse=True)
    t["item"] = inv.astype(np.int32)
    t["seen"] = t.duplicated(["uid", "item"])
    t["timestamp"] = t.ts
    return t, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--days", type=int, default=1, help="test window in days")
    ap.add_argument("--stratify", action="store_true", help="sample n_test/2 search and n_test/2 feed events")
    a = ap.parse_args()
    t, n = load()
    uid, org, seen = t.uid.to_numpy(), t.is_organic.to_numpy(), t.seen.to_numpy()
    print(f"events {len(t):,} (search share {org.mean():.3f}); users {t.uid.nunique():,}; items {n:,}; "
          f"span {pd.to_datetime(t.ts.min(), unit='s')} .. {pd.to_datetime(t.ts.max(), unit='s')}")
    print(f"repeat rate: search {seen[org == 1].mean():.3f}, feed {seen[org == 0].mean():.3f}")
    test_mask = (t.timestamp > t.timestamp.max() - a.days * 86400).to_numpy()
    if a.stratify:
        ev = np.concatenate([sample_events(test_mask & (org == f), uid, a.n_test // 2, 0) for f in (1, 0)])
    else:
        ev = sample_events(test_mask, uid, a.n_test, 0)
    prev_src = org[ev - 1]
    print(f"test events {len(ev):,}; search share {org[ev].mean():.3f}; same-source as previous event {(prev_src == org[ev]).mean():.3f}")
    res = {m: metrics(p, org[ev], seen[ev]) for m, p in run_counts(t, n, ev, test_days=a.days).items()}
    (ROOT / "runs" / f"kuaisar_pilot_counts{'_strat' if a.stratify else ''}_{a.days}d.json").write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        print(f"\n{m}   (organic = search, recommended = feed)")
        for g in GROUPS:
            x = r[g]
            print(f"  {g:<20} n={x['n']:>5}  HR@10 {x['HR@10']:.4f}  NDCG@10 {x['NDCG@10']:.4f}  HR@100 {x['HR@100']:.4f}")


if __name__ == "__main__":
    main()
