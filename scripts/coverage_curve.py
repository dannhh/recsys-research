"""Coverage vs set size over all alphas at once (pilot, step 2).

For a sample of test users, expands the whole SID tree (exact log p of every prefix and item),
then for each alpha reads off the split-conformal set at each depth from the calibration quantile.
Also reports the top-k recall curve (the oracle-free reference: a fixed-size top-k set).

Usage: python scripts/coverage_curve.py --ckpt runs/<run>/best.pt --n_test 300
"""
import argparse
import json
import time

import numpy as np
import torch

from conformal import item_node_paths, quantile, subtree_items, target_prefix_logp
from tiger import LEVELS, K, PAD, SID_OFFSET, Data, Trie, build_model, device, pad_batch

ALPHAS = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
TOPK = [10, 50, 100, 500, 1000, 2000, 5000, 10000]


@torch.no_grad()
def all_prefix_logp(model, trie, ids, mask, chunk=4096):
    """Exact log p of every node at depths 1..4 for one user. Returns list of (n_nodes_d,) arrays."""
    dev = ids.device
    enc = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    nodes = torch.zeros(1, dtype=torch.long, device=dev)
    score = torch.zeros(1, device=dev)
    toks = torch.zeros(1, 0, dtype=torch.long, device=dev)
    out_scores = []
    for l in range(LEVELS):
        n = nodes.numel()
        dec_in = torch.cat([torch.full((n, 1), PAD, device=dev, dtype=torch.long), toks], 1)
        hs = []
        for c in range(0, n, chunk):
            m = min(chunk, n - c)
            hs.append(model.decoder(input_ids=dec_in[c:c + m],
                                    encoder_hidden_states=enc.expand(m, -1, -1).contiguous(),
                                    encoder_attention_mask=mask.expand(m, -1).contiguous()).last_hidden_state[:, -1])
        logits = model.lm_head(torch.cat(hs) * (model.model_dim ** -0.5))[:, SID_OFFSET + l * K: SID_OFFSET + (l + 1) * K]
        children = trie.child_t[l][nodes]
        valid = children >= 0
        logp = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), -1)
        p_idx, c_idx = valid.nonzero(as_tuple=True)
        nodes, score = children[p_idx, c_idx], score[p_idx] + logp[p_idx, c_idx]
        toks = torch.cat([toks[p_idx], (c_idx + SID_OFFSET + l * K)[:, None]], 1)
        full = np.full(trie.child[l].max() + 1, -np.inf)
        full[nodes.cpu().numpy()] = score.cpu().numpy()
        out_scores.append(full)
    return out_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="amazon2023_Musical_Instruments"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_test", type=int, default=300)
    a = ap.parse_args()
    dev = device(); data = Data(a.data); trie = Trie(data.sids).to(dev)
    model = build_model().to(dev); model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    paths = item_node_paths(trie, data.sids); under = subtree_items(trie)

    X, Y, _ = data.eval_examples("test")
    perm = np.random.default_rng(a.seed).permutation(len(X))
    cal, test = perm[: len(X) // 2], perm[len(X) // 2:][: a.n_test]
    S = np.zeros((len(X), LEVELS))
    for i in range(0, len(X), 512):
        b = np.arange(i, min(i + 512, len(X)))
        ids, mask = pad_batch([X[j] for j in b], dev)
        S[b] = -target_prefix_logp(model, trie, ids, mask, paths[Y[b]], data.sids[Y[b]])
    Q = {al: [quantile(S[cal, d], al) for d in range(LEVELS)] for al in ALPHAS}

    nodes = {al: np.zeros((len(test), LEVELS)) for al in ALPHAS}
    items = {al: np.zeros((len(test), LEVELS)) for al in ALPHAS}
    covered = {al: np.zeros((len(test), LEVELS), dtype=bool) for al in ALPHAS}
    ranks = np.zeros(len(test), dtype=np.int64)
    t0 = time.time()
    for n, j in enumerate(test):
        ids, mask = pad_batch([X[j]], dev)
        sc = all_prefix_logp(model, trie, ids, mask)
        leaf = sc[LEVELS - 1]
        ranks[n] = int((leaf > leaf[paths[Y[j], LEVELS - 1]]).sum()) + 1
        for al in ALPHAS:
            for d in range(LEVELS):
                keep = sc[d] >= -Q[al][d]
                nodes[al][n, d] = keep.sum(); items[al][n, d] = under[d][keep].sum()
                covered[al][n, d] = keep[paths[Y[j], d]]
        if n % 50 == 0:
            print(f"{n}/{len(test)} users, {time.time() - t0:.0f}s", flush=True)

    res = {"ckpt": a.ckpt, "n_test": len(test), "sec_per_user": (time.time() - t0) / len(test),
           "topk_recall": {k: float((ranks <= k).mean()) for k in TOPK},
           "median_target_rank": float(np.median(ranks)), "conformal": {}}
    for al in ALPHAS:
        res["conformal"][str(al)] = {f"depth{d + 1}": {
            "coverage": float(covered[al][:, d].mean()),
            "coverage_all_test_by_score": float((S[perm[len(X) // 2:], d] <= Q[al][d]).mean()),
            "nodes_median": float(np.median(nodes[al][:, d])), "nodes_mean": float(nodes[al][:, d].mean()),
            "items_median": float(np.median(items[al][:, d])), "items_mean": float(items[al][:, d].mean()),
            "empty_share": float((nodes[al][:, d] == 0).mean())} for d in range(LEVELS)}
    out = a.ckpt.replace(".pt", "_curve.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"topk_recall": res["topk_recall"], "median_target_rank": res["median_target_rank"]}))
    for al in ALPHAS:
        r = res["conformal"][str(al)]
        print(f"alpha {al}: " + "  ".join(f"d{d + 1} cov {r[f'depth{d + 1}']['coverage']:.3f} "
                                          f"nodes {r[f'depth{d + 1}']['nodes_median']:.0f} items {r[f'depth{d + 1}']['items_median']:.0f}"
                                          for d in range(LEVELS)))


if __name__ == "__main__":
    main()
