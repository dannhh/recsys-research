"""Exposure-conditioned evaluation on KuaiRand-Pure: do models that rank policy exposures well also rank random ones?

For each exposure in the test window (last --days days), every model scores the exposed video given the user's click
history before the exposure time; click AUC (global) and GAUC (mean per-user AUC over users with both classes) are
computed separately on policy exposures (is_rand = 0) and random exposures (is_rand = 1). Random exposures are the
standard unbiased test in the debiasing literature; policy exposures are what logged-data evaluation measures.

Models (all fitted on clicks before the test window; SASRec also holds out one validation day):
  click popularity, exposure popularity (how often the policy showed the item: a pure policy-imitation score),
  first-order Markov, recency-weighted pairwise over the last 5 clicks, SASRec.

Usage: python scripts/kuairand_exposure_eval.py [--days 3] [--n_per_type 50000]
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score

from kuairand_pilot import D, load
from tiger import ROOT
from yambda_seq_pilot import DAY, pad_left, run_neural

W = 5


def gauc(users, y, s):
    by = defaultdict(list)
    for i, u in enumerate(users):
        by[u].append(i)
    aucs, ws = [], []
    for idx in by.values():
        yy = y[idx]
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc_score(yy, s[idx])); ws.append(len(idx))
    return float(np.average(aucs, weights=ws)), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3); ap.add_argument("--n_per_type", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    t, n = load()                                           # clicks, sorted by (uid, timestamp)
    items = np.unique(t.item_id.to_numpy())
    cut = t.timestamp.max() - a.days * DAY

    ex = pd.concat([pd.read_csv(f, usecols=["user_id", "video_id", "time_ms", "is_click", "is_rand"])
                    for f in sorted(D.glob("log_*_pure.csv"))], ignore_index=True)
    ex["ts"] = ex.time_ms // 1000
    ex["item"] = np.searchsorted(items, ex.video_id.to_numpy())
    ex = ex[(ex.item < n) & (items[np.minimum(ex.item, n - 1)] == ex.video_id)]   # exposed items that were ever clicked
    train_ex = ex[ex.ts <= cut]
    test = pd.concat([ex[(ex.ts > cut) & (ex.is_rand == r)].sample(min(a.n_per_type, ((ex.ts > cut) & (ex.is_rand == r)).sum()),
                                                                  random_state=0) for r in (0, 1)])
    print(f"test exposures {len(test):,}; click rate policy {test[test.is_rand == 0].is_click.mean():.3f}, "
          f"random {test[test.is_rand == 1].is_click.mean():.3f}")

    # click histories per user
    uid, item, ts = t.uid.to_numpy(), t.item.to_numpy(), t.timestamp.to_numpy()
    starts = np.r_[0, np.nonzero(uid[1:] != uid[:-1])[0] + 1]; ends = np.r_[starts[1:], len(t)]
    span = {uid[s]: (s, e) for s, e in zip(starts, ends)}

    def history(u, when, L):
        if u not in span:
            return np.array([], dtype=np.int64)
        s, e = span[u]
        k = s + np.searchsorted(ts[s:e], when, side="left")  # clicks strictly before the exposure
        return item[max(s, k - L):k]

    tr = ts <= cut
    rows, cols, vals = [], [], []
    for d in range(1, W + 1):
        ok = np.zeros(len(t), dtype=bool); ok[d:] = (uid[d:] == uid[:-d]) & tr[d:] & tr[:-d]
        idx = np.nonzero(ok)[0]
        rows.append(item[idx - d]); cols.append(item[idx]); vals.append(np.full(len(idx), d ** -0.5))
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr(); C.sum_duplicates()
    C1 = sp.csr_matrix((vals[0], (rows[0], cols[0])), shape=(n, n)).tocsr(); C1.sum_duplicates()
    click_pop = np.bincount(item[tr], minlength=n).astype(float)
    exp_pop = np.bincount(train_ex[train_ex.is_rand == 0].item.to_numpy(), minlength=n).astype(float)
    w = np.array([1 / math.sqrt(j + 1) for j in range(W)])

    tu, ti, tt = test.user_id.to_numpy(), test.item.to_numpy(), test.ts.to_numpy()
    S = {m: np.zeros(len(test)) for m in ("click popularity", "exposure popularity", "markov1", "pairwise5", "sasrec")}
    hists = []
    for k in range(len(test)):
        h = history(tu[k], tt[k], 200); hists.append(h)
        S["click popularity"][k] = click_pop[ti[k]]
        S["exposure popularity"][k] = exp_pop[ti[k]]
        if len(h):
            S["markov1"][k] = C1[h[-1], ti[k]]
            S["pairwise5"][k] = sum(w[j] * C[h[-1 - j], ti[k]] for j in range(min(W, len(h))))
    # tie-break count scores by click popularity so that zero-count items are still ordered
    eps = click_pop[ti] / (click_pop.max() + 1) * 1e-6
    for m in ("markov1", "pairwise5"):
        S[m] = S[m] + eps

    model, best = run_neural(t, n, np.array([], dtype=np.int64), "sasrec", a.epochs, test_days=a.days, return_model=True)
    model.eval(); dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(test), 512):
            hb = [hh if len(hh) else np.array([0]) for hh in hists[i:i + 512]]   # empty history -> dummy item
            x = pad_left(hb, 199, dev)
            h = model(x)[:, -1]
            e = model.emb.weight[torch.as_tensor(ti[i:i + 512] + 1, device=dev)]
            S["sasrec"][i:i + 512] = (h * e).sum(-1).cpu().numpy()

    y, rnd = test.is_click.to_numpy(), test.is_rand.to_numpy()
    res = {}
    print(f"\n{'model':<22}{'policy AUC':>12}{'policy GAUC':>13}{'random AUC':>12}{'random GAUC':>13}")
    for m, s in S.items():
        r = {}
        for name, mask in (("policy", rnd == 0), ("random", rnd == 1)):
            g, nu = gauc(tu[mask], y[mask], s[mask])
            r[name] = {"AUC": float(roc_auc_score(y[mask], s[mask])), "GAUC": g, "gauc_users": nu}
        res[m] = r
        print(f"{m:<22}{r['policy']['AUC']:>12.4f}{r['policy']['GAUC']:>13.4f}{r['random']['AUC']:>12.4f}{r['random']['GAUC']:>13.4f}")
    res["sasrec_best_val_HR@10"] = best
    (ROOT / "runs" / f"kuairand_exposure_eval_{a.days}d.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
