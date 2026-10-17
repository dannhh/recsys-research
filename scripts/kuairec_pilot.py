"""Second ranking-flip test: logged-policy evaluation vs fully observed evaluation on KuaiRec (Kuaishou).

KuaiRec has a policy-logged big matrix (7,176 users x 10,728 videos) and a fully observed small matrix
(1,411 users x 3,327 videos, 99.6% dense; these pairs are excluded from the big matrix). Models train on the big
matrix before the last --days days. Two evaluations of the same trained models:
  logged      next positive event in the big matrix after the cutoff (sampled), ranked over the full catalogue:
              what standard offline evaluation on logged data measures.
  observed    for each small-matrix user, rank all 3,327 small-matrix videos given the user's big-matrix history
              before the cutoff; ground truth = videos the user watched with watch_ratio >= --pos (all are exposed),
              so this approximates the true preference ranking.
Positive events (training, histories, logged targets) are watches with watch_ratio >= --pos.
Models: popularity, first-order Markov, recency-weighted pairwise (last 5), SASRec.

Usage: python scripts/kuairec_pilot.py [--days 7] [--pos 2.0]
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from tiger import ROOT
from yambda_seq_pilot import DAY, pad_left, run_counts, run_neural, sample_events

D = ROOT / "data/raw/kuairec/KuaiRec 2.0/data"
W = 5


def ndcg_recall(scores, rel, k):
    top = np.argsort(-scores)[:k]
    gains = rel[top]
    dcg = (gains / np.log2(np.arange(2, k + 2))).sum()
    ideal = (np.sort(rel)[::-1][:k] / np.log2(np.arange(2, k + 2))).sum()
    return (dcg / ideal if ideal > 0 else np.nan), (gains.sum() / rel.sum() if rel.sum() > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7); ap.add_argument("--pos", type=float, default=2.0)
    ap.add_argument("--n_test", type=int, default=10000); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pop_exclude", type=float, default=0.0, help="drop this top fraction of most popular small-matrix videos from candidates and ground truth")
    a = ap.parse_args()
    big = pd.read_csv(D / "big_matrix.csv", usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    small = pd.read_csv(D / "small_matrix.csv", usecols=["user_id", "video_id", "watch_ratio"])
    print(f"big {len(big):,}, positive share {(big.watch_ratio >= a.pos).mean():.3f}; small {len(small):,}, "
          f"positive share {(small.watch_ratio >= a.pos).mean():.3f}")
    n = int(max(big.video_id.max(), small.video_id.max())) + 1
    t = big[big.watch_ratio >= a.pos].rename(columns={"user_id": "uid", "video_id": "item"})
    t = t.assign(timestamp=t.timestamp.astype(np.int64), item=t.item.astype(np.int32), is_organic=1)
    t = t.sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)
    t["seen"] = t.duplicated(["uid", "item"])
    cut = t.timestamp.max() - a.days * DAY
    uid, item, ts = t.uid.to_numpy(), t.item.to_numpy(), t.timestamp.to_numpy()

    # ---------- logged evaluation (next positive after the cutoff, full catalogue)
    ev = sample_events(ts > cut, uid, a.n_test, 0)
    pos_counts = run_counts(t, n, ev, test_days=a.days)

    # ---------- shared count statistics for full score vectors
    tr = ts <= cut
    rows, cols, vals = [], [], []
    for d in range(1, W + 1):
        ok = np.zeros(len(t), dtype=bool); ok[d:] = (uid[d:] == uid[:-d]) & tr[d:] & tr[:-d]
        idx = np.nonzero(ok)[0]
        rows.append(item[idx - d]); cols.append(item[idx]); vals.append(np.full(len(idx), d ** -0.5))
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)); C.sum_duplicates()
    C1 = sp.csr_matrix((vals[0], (rows[0], cols[0])), shape=(n, n)); C1.sum_duplicates()
    pop = np.bincount(item[tr], minlength=n).astype(float)
    eps = pop / (pop.max() + 1) * 1e-6
    w = np.array([1 / math.sqrt(j + 1) for j in range(W)])

    model, best = run_neural(t, n, np.array([], dtype=np.int64), "sasrec", a.epochs, test_days=a.days, return_model=True)
    model.eval(); dev = next(model.parameters()).device
    E = model.emb.weight[1:].detach()

    def hist(u, upto_idx=None, when=None):
        m = uid == u
        hu, tu = item[m], ts[m]
        return hu[tu <= when] if when is not None else hu

    @torch.no_grad()
    def sas_scores(h):
        x = pad_left([h[-200:] if len(h) else np.array([0])], 199, dev)
        return (model(x)[:, -1] @ E.T)[0].cpu().numpy()

    # SASRec logged positions
    sas_pos = np.zeros(len(ev), dtype=np.int64)
    for k, e in enumerate(ev):
        start = e - 1
        while start - 1 >= 0 and uid[start - 1] == uid[e] and e - start < 200:
            start -= 1
        s = sas_scores(item[start:e])
        sas_pos[k] = (s > s[item[e]]).sum()
    logged = {m: p for m, p in pos_counts.items()}; logged["sasrec"] = sas_pos

    # ---------- logged evaluation on the SAME USERS as the observed evaluation (small-matrix users; targets are
    # necessarily non-small videos, since the big matrix excludes small-matrix pairs), full catalogue
    small_users_arr = small.user_id.unique()
    ev3 = sample_events((ts > cut) & np.isin(uid, small_users_arr), uid, a.n_test, 2)
    pos3 = run_counts(t, n, ev3, test_days=a.days)
    sp3 = np.zeros(len(ev3), dtype=np.int64)
    for k, e in enumerate(ev3):
        start = e - 1
        while start - 1 >= 0 and uid[start - 1] == uid[e] and e - start < 200:
            start -= 1
        sv = sas_scores(item[start:e]); sp3[k] = (sv > sv[item[e]]).sum()
    pos3["sasrec"] = sp3
    print(f"logged, same users as observed ({len(ev3):,} events):")
    for m, p in pos3.items():
        print(f"  {m:<12} HR@10 {(p < 10).mean():.4f}  NDCG@10 {np.where(p < 10, 1 / np.log2(p + 2), 0).mean():.4f}  HR@100 {(p < 100).mean():.4f}")

    # ---------- fully observed evaluation
    small_items = np.sort(small.video_id.unique())
    if a.pop_exclude > 0:
        order = small_items[np.argsort(-pop[small_items])]
        small_items = np.sort(order[int(len(order) * a.pop_exclude):])
        print(f"pop_exclude {a.pop_exclude}: keeping {len(small_items)} least-popular small-matrix videos")
    res_obs = {m: {"NDCG@10": [], "NDCG@50": [], "Recall@50": []} for m in ("popularity", "markov1", "pairwise5", "sasrec")}
    for u, g in small.groupby("user_id"):
        rel = np.zeros(n); rel[g.video_id.to_numpy()] = (g.watch_ratio.to_numpy() >= a.pos).astype(float)
        rel = rel[small_items]
        if rel.sum() == 0:
            continue
        h = hist(u, when=cut)
        sc = {"popularity": pop[small_items]}
        if len(h):
            sc["markov1"] = np.asarray(C1[h[-1]].todense()).ravel()[small_items] + eps[small_items]
            row = sum(w[j] * C[h[-1 - j]] for j in range(min(W, len(h))))
            sc["pairwise5"] = np.asarray(row.todense()).ravel()[small_items] + eps[small_items]
        else:
            sc["markov1"] = sc["pairwise5"] = pop[small_items]
        sc["sasrec"] = sas_scores(h)[small_items]
        for m, s in sc.items():
            n10, _ = ndcg_recall(s, rel, 10); n50, r50 = ndcg_recall(s, rel, 50)
            res_obs[m]["NDCG@10"].append(n10); res_obs[m]["NDCG@50"].append(n50); res_obs[m]["Recall@50"].append(r50)

    # ---------- logged evaluation on the SAME candidate set: non-small users, targets among small-matrix videos,
    # ranked only among the 3,327 small-matrix videos
    small_users = set(small.user_id.unique())
    in_small = np.zeros(n, dtype=bool); in_small[small_items] = True
    cand = np.nonzero((ts > cut) & in_small[item] & ~np.isin(uid, list(small_users)) & np.r_[False, uid[1:] == uid[:-1]])[0]
    ev2 = np.random.default_rng(1).choice(cand, min(a.n_test, len(cand)), replace=False)
    pos2 = {m: np.zeros(len(ev2), dtype=np.int64) for m in ("popularity", "markov1", "pairwise5", "sasrec")}
    for k, e in enumerate(ev2):
        start = e - 1
        while start - 1 >= 0 and uid[start - 1] == uid[e] and e - start < 200:
            start -= 1
        h = item[start:e]
        sc = {"popularity": pop[small_items],
              "markov1": np.asarray(C1[h[-1]].todense()).ravel()[small_items] + eps[small_items],
              "pairwise5": np.asarray(sum(w[j] * C[h[-1 - j]] for j in range(min(W, len(h)))).todense()).ravel()[small_items] + eps[small_items],
              "sasrec": sas_scores(h)[small_items]}
        tgt = np.searchsorted(small_items, item[e])
        s0 = np.nonzero(uid == uid[e])[0][0]                       # all earlier positives of this user are "seen"
        seen_mask = np.isin(small_items, item[s0:e])
        for m, sv in sc.items():
            sv = sv.astype(float).copy(); sv[seen_mask] = -np.inf; sv[tgt] = sc[m][tgt]
            pos2[m][k] = (sv > sv[tgt]).sum()
    print(f"logged-on-small-candidates events {len(ev2):,}")

    out = {"best_val_HR@10": best, "logged": {}, "observed": {}, "logged_small_candidates": {}}
    for m, p in pos2.items():
        out["logged_small_candidates"][m] = {"HR@10": float((p < 10).mean()), "NDCG@10": float(np.where(p < 10, 1 / np.log2(p + 2), 0).mean()),
                                              "HR@100": float((p < 100).mean())}
        print(f"  logged, small candidates  {m:<12} HR@10 {out['logged_small_candidates'][m]['HR@10']:.4f}  NDCG@10 {out['logged_small_candidates'][m]['NDCG@10']:.4f}  HR@100 {out['logged_small_candidates'][m]['HR@100']:.4f}")
    print(f"\n{'model':<12}{'logged HR@10':>14}{'logged NDCG@10':>16}{'logged HR@100':>15} |{'obs NDCG@10':>13}{'obs NDCG@50':>13}{'obs Recall@50':>15}")
    for m in ("popularity", "markov1", "pairwise5", "sasrec"):
        p = logged[m]
        lg = {"HR@10": float((p < 10).mean()), "NDCG@10": float(np.where(p < 10, 1 / np.log2(p + 2), 0).mean()), "HR@100": float((p < 100).mean())}
        ob = {k: float(np.nanmean(v)) for k, v in res_obs[m].items()}
        out["logged"][m], out["observed"][m] = lg, ob
        print(f"{m:<12}{lg['HR@10']:>14.4f}{lg['NDCG@10']:>16.4f}{lg['HR@100']:>15.4f} |{ob['NDCG@10']:>13.4f}{ob['NDCG@50']:>13.4f}{ob['Recall@50']:>15.4f}")
    (ROOT / "runs" / f"kuairec_pilot_{a.days}d_pos{a.pos}_popex{a.pop_exclude}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
