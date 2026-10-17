"""Robustness runs for the organic-vs-recommended claim (Yambda-50M, step 1 of plan-organic-vs-recommended).

Extends yambda_seq_pilot.py: one training cutoff, several test days, tunable model size, and saved per-event ranks
so that groups, days and models can be compared later with paired bootstraps.
  - The last `test_days` days are test days. `n_test` events are sampled from each test day (seed 0, fixed across runs).
  - Neural models train on events before the validation day (the day before the first test day) and early-stop on it.
  - Count models use all events before the first test day.
  - Test-time history is the full log before each test event, as in the pilot.

Output: <runs>/yambda_bench/<tag>.json (metrics per day and pooled) and <tag>.npz (ranks, group labels, day).

Usage: python scripts/yambda_bench.py --model sasrec|gru4rec|counts [--d 64] [--lr 1e-3] [--seed 0] [--tag name]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import yambda_seq_pilot as P

RUNS = Path(os.environ.get("RUNS_DIR", P.ROOT / "runs")) / "yambda_bench"


def test_events(t, test_days, n_test):
    uid, ts = t.uid.to_numpy(), t.timestamp.to_numpy()
    T = ts.max()
    evs, days = [], []
    for k in range(test_days):                                           # day 0 = first test day
        lo, hi = T - (test_days - k) * P.DAY, T - (test_days - k - 1) * P.DAY
        ev = P.sample_events((ts > lo) & (ts <= hi), uid, n_test, k)
        evs.append(np.sort(ev)); days.append(np.full(len(ev), k, np.int8))
    return np.concatenate(evs), np.concatenate(days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sasrec", "gru4rec", "counts"])
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n_val", type=int, default=10000)
    ap.add_argument("--test_days", type=int, default=3)
    ap.add_argument("--n_test", type=int, default=10000, help="per test day")
    ap.add_argument("--max_minutes", type=float, default=0, help="training time budget, 0 = none")
    ap.add_argument("--max_windows", type=int, default=0, help="smoke test: cap training windows")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or (a.model if a.model == "counts" else f"{a.model}_d{a.d}_l{a.layers}_lr{a.lr:g}_s{a.seed}")
    RUNS.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    t, n = P.load()
    org, seen = t.is_organic.to_numpy(), t.seen.to_numpy()
    ev, day = test_events(t, a.test_days, a.n_test)
    print(f"[{tag}] listen+ {len(t):,}; items {n:,}; test events {len(ev):,} over {a.test_days} days "
          f"(organic {org[ev].mean():.3f}); load {time.time() - t0:.0f}s", flush=True)

    info = {"args": vars(a), "n_items": n}
    if a.model == "counts":
        pos = P.run_counts(t, n, ev, test_days=a.test_days)
    else:
        if a.max_windows:
            _w = P.windows
            P.windows = lambda seqs, L: _w(seqs, L)[:a.max_windows]       # noqa: E731
        p, best = P.run_neural(t, n, ev, a.model, a.epochs, lr=a.lr, seed=a.seed, test_days=a.test_days,
                               d=a.d, layers=a.layers, n_val=a.n_val, patience=a.patience, amp=True,
                               max_minutes=a.max_minutes)
        pos = {tag: p}
        info["best_val_HR@10"] = best
    info["minutes"] = (time.time() - t0) / 60

    res = {}
    for m, p in pos.items():
        res[m] = {"pooled": P.metrics(p, org[ev], seen[ev])}
        for k in range(a.test_days):
            s = day == k
            res[m][f"day{k}"] = P.metrics(p[s], org[ev][s], seen[ev][s])
    (RUNS / f"{tag}.json").write_text(json.dumps({"info": info, "results": res}, indent=1))
    np.savez_compressed(RUNS / f"{tag}.npz", ev=ev, day=day, org=org[ev], seen=seen[ev],
                        **{f"pos_{m}": p for m, p in pos.items()})
    for m, r in res.items():
        print(f"\n[{tag}] {m} pooled")
        for g in P.GROUPS:
            x = r["pooled"][g]
            print(f"  {g:<20} n={x['n']:>6}  HR@10 {x['HR@10']:.4f}  NDCG@10 {x['NDCG@10']:.4f}  HR@100 {x['HR@100']:.4f}")
    print(f"[{tag}] done in {info['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
