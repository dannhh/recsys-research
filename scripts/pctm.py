"""Pairwise capacity probe (PCTM, Petrov et al., RecSys 2026, arXiv 2608.19833) on our leave-one-out split.

Question: can a model that only sums recency-weighted pairwise transitions match generative recommenders
(our TIGER; published SPRINT / DiffGRM / EPIC) on Amazon 2023 Instruments?

PCTM (their Algorithms 1-2):
  C_ba = sum_{d<=W} gamma_d * #(a appears d positions after b)          counts from training sequences
  s(a | h) = sum_j w_j * log(1 + |I| C_{h_j a} / tau) + lam * log p_pop(a)  j = 1 is the newest history item
History weights w are exponential (E(r)) or head-tail (H(k, rho, r)), normalised to sum to one.
Counts use only the users' train sequences, the same data TIGER trains on. Hyperparameters are chosen on the
valid target (history = train); the test target uses history = train + valid. Items already in the history
are excluded from the ranking (targets never repeat in this split, so this only removes impossible items).

Usage: python scripts/pctm.py [--n_search 60]
"""
import argparse
import itertools
import json
import time

import numpy as np
import scipy.sparse as sp

from tiger import ROOT, Data, rank_metrics


def counts(seqs, n, W, gamma):
    rows, cols, vals = [], [], []
    for s in seqs:
        s = np.asarray(s)
        for d in range(1, min(W, len(s) - 1) + 1):
            rows.append(s[:-d]); cols.append(s[d:]); vals.append(np.full(len(s) - d, gamma(d)))
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    C.sum_duplicates()
    return C


def kernel(kind, L):
    j = np.arange(L)
    if kind[0] == "E":
        w = kind[1] ** j
    else:  # head-tail H(k, rho, r)
        _, k, rho, r = kind
        if L <= k:
            w = r ** j
        else:
            head = r ** j[:k]
            w = np.concatenate([rho * head / head.sum(), np.full(L - k, (1 - rho) / (L - k))])
    return w / w.sum()


def score_topk(C, pop, hists, n, tau, kind, Lmax, lam, k=10, bs=4096):
    Lg = C.copy(); Lg.data = np.log1p(n * Lg.data / tau)
    logpop = lam * np.log(pop)
    out = np.zeros((len(hists), k), dtype=np.int64)
    for i in range(0, len(hists), bs):
        rows, cols, vals = [], [], []
        for r, h in enumerate(hists[i:i + bs]):
            h = h[::-1][:Lmax]                                  # newest first
            rows += [r] * len(h); cols += list(h); vals += list(kernel(kind, len(h)))
        Wh = sp.csr_matrix((vals, (rows, cols)), shape=(len(hists[i:i + bs]), n))
        S = (Wh @ Lg).toarray() + logpop
        for r, h in enumerate(hists[i:i + bs]):
            S[r, h] = -np.inf
        top = np.argpartition(-S, k, 1)[:, :k]
        out[i:i + len(top)] = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="amazon2023_Musical_Instruments")
    ap.add_argument("--n_search", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = Data(a.data); n = data.n_items; loo = list(data.loo.values())
    train = [r["train"] for r in loo]
    freq = np.bincount(np.concatenate(train), minlength=n)
    pop = (freq + 1) / (freq.sum() + n)
    h_val, y_val = [r["train"] for r in loo], np.array([r["valid"] for r in loo])
    h_test, y_test = [r["train"] + [r["valid"]] for r in loo], np.array([r["test"] for r in loo])

    gammas = {"1/d": lambda d: 1 / d, "1/sqrt(d)": lambda d: d ** -0.5, "0.7^(d-1)": lambda d: 0.7 ** (d - 1),
              "1": lambda d: 1.0}
    grid = list(itertools.product(gammas, [1, 3, 10, 20], [10, 100, 1000, 5000, 20000],
                                  [("E", 0.3), ("E", 0.6), ("E", 0.8), ("E", 1.0), ("H", 7, 0.8, 0.6), ("H", 3, 0.8, 0.5)],
                                  [0.0, 0.1, 0.3, 0.6]))
    rng = np.random.default_rng(a.seed)
    cand = [grid[i] for i in rng.choice(len(grid), a.n_search, replace=False)]
    cand.append(("1", 1, 10, ("E", 1.0), 0.0))                  # first-order Markov control (last item only below)
    Cs = {}
    t0 = time.time(); results = []
    for g, W, tau, kind, lam in cand:
        if (g, W) not in Cs:
            Cs[(g, W)] = counts(train, n, W, gammas[g])
        Lmax = 1 if (g, W, kind) == ("1", 1, ("E", 1.0)) else 50
        m = rank_metrics(score_topk(Cs[(g, W)], pop, h_val, n, tau, kind, Lmax, lam), y_val)
        results.append({"gamma": g, "W": W, "tau": tau, "kernel": list(kind), "Lmax": Lmax, "lam": lam, **m})
        print(f"{g:>10} W{W:<3} tau{tau:<6} {str(kind):<20} lam{lam:<4} valid R@10 {m['R@10']:.4f} N@10 {m['N@10']:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    results.sort(key=lambda r: -r["N@10"])
    best = results[0]
    markov = [r for r in results if r["Lmax"] == 1][0]
    test = {}
    for name, r in (("pctm_best", best), ("markov_last_item", markov)):
        top = score_topk(Cs[(r["gamma"], r["W"])], pop, h_test, n, r["tau"], tuple(r["kernel"]), r["Lmax"], r["lam"])
        test[name] = {"config": r, "test": rank_metrics(top, y_test)}
    pop_top = np.argsort(-freq)[:10]
    test["popularity"] = {"test": rank_metrics(np.tile(pop_top, (len(y_test), 1)), y_test)}
    out = ROOT / "runs" / f"{a.data}_pctm.json"
    out.write_text(json.dumps({"search": results, "test": test}, indent=1))
    print(json.dumps({k: v["test"] for k, v in test.items()}, indent=1))
    print("best config:", {k: best[k] for k in ("gamma", "W", "tau", "kernel", "Lmax", "lam")})


if __name__ == "__main__":
    main()
