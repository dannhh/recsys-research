"""Split-conformal prediction sets over the SID tree for a trained TIGER model (pilot, step 1).

Sets compared, each calibrated by split conformal on a path score (valid marginal coverage):
  depth d in {1, 2, 3}: prefix sets  C_d(x) = {prefixes v at depth d : -log p(v | x) <= q_d}
  depth 4:              item sets    C_4(x) = {items i : -log p(i | x) <= q_4}
p(v | x) is the exact trie-constrained model probability of the prefix (product of renormalised
token probabilities along the path), so the true target's score comes from teacher forcing and
sets are found by exact threshold search (keep every node whose prefix score is under q_d;
a prefix score can only decrease along a path, so pruning is exact).

Reported per depth: coverage, nodes returned, items covered, search cost (nodes scored), and the
share of users whose search hit the node cap (then the set is truncated and coverage can drop).

Usage: python scripts/conformal.py --ckpt runs/<run>/epoch5.pt --alpha 0.1 0.2
"""
import argparse
import json
import time

import numpy as np
import torch

from tiger import LEVELS, K, PAD, SID_OFFSET, Data, Trie, build_model, device, pad_batch


def item_node_paths(trie, sids):
    """(n_items, LEVELS) node id of each item's prefix at depths 1..4."""
    paths = np.zeros((len(sids), LEVELS), dtype=np.int64)
    node = np.zeros(len(sids), dtype=np.int64)
    for l in range(LEVELS):
        node = trie.child[l][node, sids[:, l]]
        paths[:, l] = node
    return paths


def subtree_items(trie):
    """items_under[d][v]: number of items under node v at depth d+1."""
    counts = [None] * LEVELS
    counts[LEVELS - 1] = np.ones(len(trie.leaf_item), dtype=np.int64)
    for l in range(LEVELS - 1, 0, -1):
        ch = trie.child[l]  # parents at depth l (nodes of level l), children at depth l+1
        c = np.zeros(ch.shape[0], dtype=np.int64)
        mask = ch >= 0
        c += np.where(mask, counts[l][ch.clip(0)], 0).sum(1)
        counts[l - 1] = c
    return counts


@torch.no_grad()
def target_prefix_logp(model, trie, ids, mask, paths, codes):
    """Exact log p(target prefix of depth d | x) for d = 1..4 by teacher forcing. Returns (B, 4)."""
    dev = ids.device
    tok = torch.as_tensor(codes + SID_OFFSET + np.arange(LEVELS) * K, device=dev)
    dec_in = torch.cat([torch.full((len(codes), 1), PAD, device=dev, dtype=torch.long), tok[:, :-1]], 1)
    out = model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec_in).logits  # (B, 4, V)
    lp = []
    parent = torch.zeros(len(codes), dtype=torch.long, device=dev)
    for l in range(LEVELS):
        logits = out[:, l, SID_OFFSET + l * K: SID_OFFSET + (l + 1) * K]
        valid = trie.child_t[l][parent] >= 0
        logp = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), -1)
        lp.append(logp.gather(1, torch.as_tensor(codes[:, l:l + 1], device=dev)).squeeze(1))
        parent = torch.as_tensor(paths[:, l], device=dev)
    return torch.cumsum(torch.stack(lp, 1), 1).cpu().numpy()


@torch.no_grad()
def threshold_search(model, trie, ids, mask, thr, cap):
    """For one user: all nodes at each depth with log p >= -thr[d]. Returns per-depth node ids, cost, capped."""
    dev = ids.device
    enc = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    nodes = torch.zeros(1, dtype=torch.long, device=dev)
    score = torch.zeros(1, device=dev)
    toks = torch.zeros(1, 0, dtype=torch.long, device=dev)
    kept, cost, capped = [], 0, False
    for l in range(LEVELS):
        n = nodes.numel()
        if n == 0:
            kept.append(np.array([], dtype=np.int64)); continue
        dec_in = torch.cat([torch.full((n, 1), PAD, device=dev, dtype=torch.long), toks], 1)
        outs = []
        for c in range(0, n, 2048):  # chunked, contiguous copies: MPS asserts on large expanded views
            m = min(2048, n - c)
            outs.append(model.decoder(input_ids=dec_in[c:c + m],
                                      encoder_hidden_states=enc.expand(m, -1, -1).contiguous(),
                                      encoder_attention_mask=mask.expand(m, -1).contiguous()).last_hidden_state[:, -1])
        out = torch.cat(outs)
        logits = model.lm_head(out * (model.model_dim ** -0.5))[:, SID_OFFSET + l * K: SID_OFFSET + (l + 1) * K]
        children = trie.child_t[l][nodes]
        valid = children >= 0
        logp = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), -1)
        cand = score[:, None] + logp
        cost += int(valid.sum())
        # deepest threshold still to be served decides pruning: keep if within any remaining depth's threshold
        keep = cand >= -max(thr[l:])
        p_idx, c_idx = keep.nonzero(as_tuple=True)
        if p_idx.numel() > cap:
            capped = True
            top = cand[p_idx, c_idx].topk(cap).indices
            p_idx, c_idx = p_idx[top], c_idx[top]
        nodes, score = children[p_idx, c_idx], cand[p_idx, c_idx]
        toks = torch.cat([toks[p_idx], (c_idx + SID_OFFSET + l * K)[:, None]], 1)
        kept.append(nodes[score >= -thr[l]].cpu().numpy())
    return kept, cost, capped


def quantile(scores, alpha):
    n = len(scores)
    return float(np.sort(scores)[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="amazon2023_Musical_Instruments"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.2]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=3000); ap.add_argument("--cap", type=int, default=20000)
    a = ap.parse_args()
    dev = device(); data = Data(a.data); trie = Trie(data.sids).to(dev)
    model = build_model().to(dev); model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    paths = item_node_paths(trie, data.sids); under = subtree_items(trie)

    X, Y, _ = data.eval_examples("test")
    perm = np.random.default_rng(a.seed).permutation(len(X))
    cal, test = perm[: len(X) // 2], perm[len(X) // 2:][: a.n_test]

    t0 = time.time(); S = np.zeros((len(X), LEVELS))
    for i in range(0, len(X), 512):
        b = np.arange(i, min(i + 512, len(X)))
        ids, mask = pad_batch([X[j] for j in b], dev)
        S[b] = -target_prefix_logp(model, trie, ids, mask, paths[Y[b]], data.sids[Y[b]])
    print(f"scored {len(X):,} users in {time.time() - t0:.0f}s; median -log p(item) {np.median(S[:, 3]):.2f}")

    results = {"ckpt": a.ckpt, "n_cal": len(cal), "n_test": len(test), "alpha": {}}
    for alpha in a.alpha:
        q = [quantile(S[cal, d], alpha) for d in range(LEVELS)]
        stats = {d: {"covered": [], "nodes": [], "items": [], } for d in range(LEVELS)}
        costs, capped = [], 0
        t0 = time.time()
        for j in test:
            ids, mask = pad_batch([X[j]], dev)
            kept, cost, cap_hit = threshold_search(model, trie, ids, mask, q, a.cap)
            costs.append(cost); capped += cap_hit
            for d in range(LEVELS):
                stats[d]["covered"].append(paths[Y[j], d] in set(kept[d].tolist()))
                stats[d]["nodes"].append(len(kept[d]))
                stats[d]["items"].append(int(under[d][kept[d]].sum()) if len(kept[d]) else 0)
        res = {"q": q, "search_sec": round(time.time() - t0), "cost_median": float(np.median(costs)),
               "cost_p99": float(np.percentile(costs, 99)), "capped_share": capped / len(test)}
        for d in range(LEVELS):
            s = stats[d]
            res[f"depth{d + 1}"] = {"coverage": float(np.mean(s["covered"])),
                                    "coverage_by_score": float(np.mean(S[test, d] <= q[d])),
                                    "nodes_mean": float(np.mean(s["nodes"])), "nodes_median": float(np.median(s["nodes"])),
                                    "items_mean": float(np.mean(s["items"])), "items_median": float(np.median(s["items"]))}
        results["alpha"][str(alpha)] = res
        print(json.dumps({"alpha": alpha, **res}, indent=1), flush=True)
    out = a.ckpt.replace(".pt", f"_conformal.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
