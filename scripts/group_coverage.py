"""Group-conditional coverage of candidate sets (pilot, step 3).

Question: where a fixed top-k set under-covers (tail items, short-history users), can conformal
sets keep coverage per group, and at what size?

Groups (leave-one-out split):
  item popularity: train count of the target item (tail <= 5, 6-25, 26-100, head > 100)
  user activity:   history length (<= 4, 5-9, >= 10)

Methods, each calibrated to the same target coverage 1 - alpha:
  topk        fixed k, k = conformal quantile of the target rank (ranks from expanded cal users)
  marginal    {i : -log p(i|x) <= q}, one q (split conformal)
  mondrian_u  one q per user-activity group (valid per user group)
  classcond_i one q per item-popularity group, applied to the items of that group (valid per item group)

Leaf scores of expanded users are cached in <ckpt>_leaf_<n>.npy.
Usage: python scripts/group_coverage.py --ckpt runs/<run>/best.pt --n_expand 2000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from conformal import item_node_paths, quantile, target_prefix_logp
from coverage_curve import all_prefix_logp
from tiger import LEVELS, Data, Trie, build_model, device, pad_batch

ALPHAS = [0.7, 0.5, 0.3]
POP_BINS = [(0, 5, "tail<=5"), (6, 25, "6-25"), (26, 100, "26-100"), (101, 10**9, "head>100")]
ACT_BINS = [(0, 4, "hist<=4"), (5, 9, "hist5-9"), (10, 10**9, "hist>=10")]


def bin_of(v, bins):
    out = np.full(len(v), -1)
    for g, (lo, hi, _) in enumerate(bins):
        out[(v >= lo) & (v <= hi)] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="amazon2023_Musical_Instruments"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_expand", type=int, default=2000)
    a = ap.parse_args()
    dev = device(); data = Data(a.data); trie = Trie(data.sids).to(dev)
    model = build_model().to(dev); model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    paths = item_node_paths(trie, data.sids)

    X, Y, U = data.eval_examples("test")
    cnt = np.zeros(data.n_items, dtype=np.int64)
    for r in data.loo.values():
        np.add.at(cnt, r["train"], 1)
    hist_len = np.array([len(data.loo[u]["train"]) + 1 for u in U])
    item_pop_g = bin_of(cnt, POP_BINS)                     # per item
    user_pop_g = item_pop_g[Y]                              # per user: group of its target
    user_act_g = bin_of(hist_len, ACT_BINS)

    perm = np.random.default_rng(a.seed).permutation(len(X))
    cal, rest = perm[: len(X) // 2], perm[len(X) // 2:]
    exp_cal, test = rest[: a.n_expand // 2], rest[a.n_expand // 2: a.n_expand]  # rank-calibration users, test users

    S = np.zeros(len(X))
    for i in range(0, len(X), 512):
        b = np.arange(i, min(i + 512, len(X)))
        ids, mask = pad_batch([X[j] for j in b], dev)
        S[b] = -target_prefix_logp(model, trie, ids, mask, paths[Y[b]], data.sids[Y[b]])[:, LEVELS - 1]

    cache = Path(a.ckpt.replace(".pt", f"_leaf_{a.n_expand}.npy"))
    exp_users = np.concatenate([exp_cal, test])
    if cache.exists():
        leaf = np.load(cache)
    else:
        leaf = np.zeros((len(exp_users), len(trie.leaf_item)), dtype=np.float32)
        t0 = time.time()
        for n, j in enumerate(exp_users):
            ids, mask = pad_batch([X[j]], dev)
            leaf[n] = all_prefix_logp(model, trie, ids, mask)[LEVELS - 1]
            if n % 100 == 0:
                print(f"expanded {n}/{len(exp_users)}, {time.time() - t0:.0f}s", flush=True)
        np.save(cache, leaf)
    leaf_pop_g = item_pop_g[trie.leaf_item]                 # popularity group of each leaf
    tgt_leaf = paths[Y[exp_users], LEVELS - 1]
    tgt_score = leaf[np.arange(len(exp_users)), tgt_leaf]
    ranks = (leaf > tgt_score[:, None]).sum(1) + 1
    n_cal = len(exp_cal); T = slice(n_cal, None)            # rows of test users in `leaf`

    res = {"ckpt": a.ckpt, "n_cal": len(cal), "n_rank_cal": n_cal, "n_test": len(test),
           "groups": {"pop": [b[2] for b in POP_BINS], "act": [b[2] for b in ACT_BINS]},
           "group_share_test": {"pop": [float((user_pop_g[test] == g).mean()) for g in range(len(POP_BINS))],
                                "act": [float((user_act_g[test] == g).mean()) for g in range(len(ACT_BINS))]},
           "alpha": {}}
    for al in ALPHAS:
        k = int(quantile(ranks[:n_cal].astype(float), al))
        q = quantile(S[cal], al)
        q_u = np.array([quantile(S[cal][user_act_g[cal] == g], al) for g in range(len(ACT_BINS))])
        q_i = np.array([quantile(S[cal][user_pop_g[cal] == g], al) for g in range(len(POP_BINS))])
        L = -leaf[T]                                         # (n_test, n_leaves) scores, lower = more likely
        sets = {
            "topk": (ranks[T] <= k, np.full(len(test), k)),
            "marginal": (tgt_score[T] >= -q, (L <= q).sum(1)),
            "mondrian_u": (-tgt_score[T] <= q_u[user_act_g[test]], (L <= q_u[user_act_g[test]][:, None]).sum(1)),
            "classcond_i": (-tgt_score[T] <= q_i[user_pop_g[test]], (L <= q_i[leaf_pop_g][None, :]).sum(1)),
        }
        out = {"k": k, "q": q, "q_user_groups": q_u.tolist(), "q_item_groups": q_i.tolist()}
        for name, (cov, size) in sets.items():
            m = {"coverage": float(cov.mean()), "size_mean": float(size.mean()), "size_median": float(np.median(size)),
                 "size_p10_p90": np.percentile(size, [10, 90]).tolist()}
            for gname, gid, bins in (("pop", user_pop_g[test], POP_BINS), ("act", user_act_g[test], ACT_BINS)):
                m[gname] = {bins[g][2]: {"coverage": float(cov[gid == g].mean()), "size_mean": float(size[gid == g].mean()),
                                         "n": int((gid == g).sum())} for g in range(len(bins))}
            out[name] = m
        res["alpha"][str(al)] = out

    out_path = a.ckpt.replace(".pt", "_groups.json")
    Path(out_path).write_text(json.dumps(res, indent=1))
    for al, o in res["alpha"].items():
        print(f"\n=== target coverage {1 - float(al):.1f}  (k={o['k']})")
        for name in ("topk", "marginal", "mondrian_u", "classcond_i"):
            m = o[name]
            pop = " ".join(f"{g} {v['coverage']:.2f}" for g, v in m["pop"].items())
            act = " ".join(f"{g} {v['coverage']:.2f}" for g, v in m["act"].items())
            print(f"{name:12s} cov {m['coverage']:.3f} size mean {m['size_mean']:.0f} med {m['size_median']:.0f} | {pop} | {act}")


if __name__ == "__main__":
    main()
