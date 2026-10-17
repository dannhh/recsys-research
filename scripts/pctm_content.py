"""Content-aware pairwise probe (step 2 of the probe study).

Question: is the margin of the newest generative recommenders over PCTM explained by item content (which SIDs carry
and PCTM lacks) rather than by higher-order sequence modelling?

Every addition stays pairwise (the score is still a weighted sum over single history items):
  source smoothing  C~ = C + beta * S C       a rare source item borrows the transitions of its text neighbours
  content expert    + mu * sum_j w_j S[h_j, a]  a candidate scores by text similarity to each history item
S is the k-nearest-neighbour graph of the Sentence-T5 item embeddings (cosine, self excluded).
PCTM hyperparameters are fixed at the best plain-PCTM config; beta, mu, k are tuned on the valid target.
Test results are broken down by target popularity (train count), as in group_coverage.py.

Usage: python scripts/pctm_content.py
"""
import itertools
import json
import time

import numpy as np
import scipy.sparse as sp

from pctm import counts, kernel
from tiger import ROOT, Data, rank_metrics

DATA = "amazon2023_Musical_Instruments"
GAMMA, W, TAU, KIND, LMAX = (lambda d: d ** -0.5), 20, 50000, ("H", 10, 0.8, 0.7), 50
POP_BINS = [(0, 5, "tail<=5"), (6, 25, "6-25"), (26, 100, "26-100"), (101, 10**9, "head>100")]


def knn_graph(emb, k, bs=2048):
    n = len(emb); rows, cols, vals = [], [], []
    for i in range(0, n, bs):
        s = emb[i:i + bs] @ emb.T
        s[np.arange(len(s)), np.arange(i, i + len(s))] = -1
        top = np.argpartition(-s, k, 1)[:, :k]
        rows.append(np.repeat(np.arange(i, i + len(s)), k)); cols.append(top.ravel())
        vals.append(np.clip(np.take_along_axis(s, top, 1).ravel(), 0, None))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def history_matrix(hists, n):
    rows, cols, vals = [], [], []
    for r, h in enumerate(hists):
        h = h[::-1][:LMAX]
        rows += [r] * len(h); cols += list(h); vals += list(kernel(KIND, len(h)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(hists), n))


def topk(C, S, mu, hists, n, k=10, bs=4096):
    Lg = C.copy(); Lg.data = np.log1p(n * Lg.data / TAU)
    out = np.zeros((len(hists), k), dtype=np.int64)
    for i in range(0, len(hists), bs):
        Wh = history_matrix(hists[i:i + bs], n)
        M = (Wh @ Lg).toarray()
        if mu:
            M += mu * (Wh @ S).toarray()
        for r, h in enumerate(hists[i:i + bs]):
            M[r, h] = -np.inf
        top = np.argpartition(-M, k, 1)[:, :k]
        out[i:i + len(top)] = np.take_along_axis(top, np.argsort(-np.take_along_axis(M, top, 1), 1), 1)
    return out


def by_group(pred, y, cnt):
    g = np.digitize(cnt[y], [b[1] + 0.5 for b in POP_BINS[:-1]])
    return {POP_BINS[i][2]: {**rank_metrics(pred[g == i], y[g == i]), "n": int((g == i).sum())} for i in range(len(POP_BINS))}


def main():
    data = Data(DATA); n = data.n_items; loo = list(data.loo.values())
    train = [r["train"] for r in loo]
    cnt = np.bincount(np.concatenate(train), minlength=n)
    hv, yv = [r["train"] for r in loo], np.array([r["valid"] for r in loo])
    ht, yt = [r["train"] + [r["valid"]] for r in loo], np.array([r["test"] for r in loo])
    emb = np.load(ROOT / "data" / "processed" / DATA / "emb.npy")
    C0 = counts(train, n, W, GAMMA)

    results, t0 = [], time.time()
    graphs = {k: knn_graph(emb, k) for k in (10, 50)}
    for k, beta, mu in itertools.product((10, 50), (0, 0.1, 0.3, 1.0), (0, 0.3, 1, 3)):
        if k == 50 and beta == 0 and mu == 0:
            continue
        S = graphs[k]
        C = C0 + beta * (S @ C0) if beta else C0
        m = rank_metrics(topk(C, S, mu, hv, n), yv)
        results.append({"k": k, "beta": beta, "mu": mu, **m})
        print(f"k{k:<3} beta{beta:<4} mu{mu:<4} valid R@10 {m['R@10']:.4f} N@10 {m['N@10']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    results.sort(key=lambda r: -r["N@10"])
    best = results[0]
    test = {}
    for name, (k, beta, mu) in (("pctm", (10, 0, 0)), ("pctm_content", (best["k"], best["beta"], best["mu"]))):
        S = graphs[k]; C = C0 + beta * (S @ C0) if beta else C0
        pred = topk(C, S, mu, ht, n)
        test[name] = {"config": {"k": k, "beta": beta, "mu": mu}, "test": rank_metrics(pred, yt), "by_pop": by_group(pred, yt, cnt)}
    (ROOT / "runs" / f"{DATA}_pctm_content.json").write_text(json.dumps({"search": results, "test": test}, indent=1))
    for name, v in test.items():
        print(name, v["config"], {m: round(x, 4) for m, x in v["test"].items()})
        print("   ", {g: (round(x["R@10"], 4), round(x["N@10"], 4), x["n"]) for g, x in v["by_pop"].items()})


if __name__ == "__main__":
    main()
