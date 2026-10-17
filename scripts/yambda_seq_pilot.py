"""Decisive pilot: do model rankings change across event groups? (Yambda-50M, step 1 of plan-organic-vs-recommended)

Same data and test events as yambda_organic_pilot.py (listen+, last day is test, the same 10,000 sampled test events
for the default seed), now with neural sequential models next to the count models:
  popularity, first-order Markov, recency-weighted pairwise (last 5), GRU4Rec, SASRec.
Neural models use sampled softmax: K shared negatives per batch drawn from count^0.75 with logQ correction,
accidental hits masked. They train on events before the validation day (the day before test) and early-stop on
validation HR@10. Count models use all events before the test day, as in the earlier pilot.
Evaluation ranks the full catalogue (no seen-item filtering) and reports HR@10, NDCG@10, HR@100 per group:
target organic/recommended x repeat/new.

Usage: python scripts/yambda_seq_pilot.py --model sasrec|gru4rec|counts [--epochs 8] [--n_test 10000]
"""
import argparse
import json
import math
import time

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiger import ROOT, device

DAY = 86400


def load():
    t = pq.read_table(ROOT / "data/raw/yambda/listens_50m.parquet",
                      columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"]).to_pandas()
    t = t[t.played_ratio_pct >= 50].sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)
    items, inv = np.unique(t.item_id.to_numpy(), return_inverse=True)
    t["item"] = inv.astype(np.int32)
    t["seen"] = t.duplicated(["uid", "item"])
    return t, len(items)


def sample_events(mask, uid, n, seed):
    prev_ok = np.r_[False, uid[1:] == uid[:-1]]
    cand = np.nonzero(mask & prev_ok)[0]
    return np.random.default_rng(seed).choice(cand, min(n, len(cand)), replace=False)


GROUPS = ["all", "organic", "recommended", "organic, repeat", "organic, new", "recommended, repeat", "recommended, new"]


def group_masks(org, seen):
    return {"all": np.ones(len(org), bool), "organic": org == 1, "recommended": org == 0,
            "organic, repeat": (org == 1) & seen, "organic, new": (org == 1) & ~seen,
            "recommended, repeat": (org == 0) & seen, "recommended, new": (org == 0) & ~seen}


def metrics(pos, org, seen):
    out = {}
    for g, m in group_masks(org, seen).items():
        p = pos[m]
        out[g] = {"n": int(m.sum()), "HR@10": float((p < 10).mean()), "HR@100": float((p < 100).mean()),
                  "NDCG@10": float(np.where(p < 10, 1 / np.log2(p + 2), 0).mean())}
    return out


# ------------------------------------------------------------------ count models
def run_counts(t, n, ev, W=5, test_days=1):
    uid, item = t.uid.to_numpy(), t.item.to_numpy()
    tr = (t.timestamp <= t.timestamp.max() - test_days * DAY).to_numpy()   # must match the test window
    rows, cols, vals = [], [], []
    for d in range(1, W + 1):
        ok = np.zeros(len(t), dtype=bool); ok[d:] = (uid[d:] == uid[:-d]) & tr[d:] & tr[:-d]
        idx = np.nonzero(ok)[0]
        rows.append(item[idx - d]); cols.append(item[idx]); vals.append(np.full(len(idx), d ** -0.5, dtype=np.float32))
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)); C.sum_duplicates()
    C1 = sp.csr_matrix((vals[0], (rows[0], cols[0])), shape=(n, n)); C1.sum_duplicates()
    pop_top = np.argsort(-np.bincount(item[tr], minlength=n))[:100]
    w = np.array([1 / math.sqrt(j + 1) for j in range(W)]); w /= w.sum()

    def pos_of(row, tgt):
        r = row.tocoo()
        order = r.col[np.argsort(-r.data)][:100] if r.nnz else np.array([], dtype=np.int64)
        order = np.concatenate([order, pop_top[~np.isin(pop_top, order)]])[:100]
        hit = np.nonzero(order == tgt)[0]
        return hit[0] if len(hit) else 10**9

    pos = {m: np.full(len(ev), 10**9) for m in ("popularity", "markov1", "pairwise5")}
    for k, e in enumerate(ev):
        tgt = item[e]
        hit = np.nonzero(pop_top == tgt)[0]
        pos["popularity"][k] = hit[0] if len(hit) else 10**9
        pos["markov1"][k] = pos_of(C1[item[e - 1]], tgt)
        row = None
        for jj, j in enumerate(j for j in range(e - 1, max(e - 1 - W, -1), -1) if uid[j] == uid[e]):
            r = C[item[j]] * w[jj]
            row = r if row is None else row + r
        pos["pairwise5"][k] = pos_of(row, tgt)
    return pos


# ------------------------------------------------------------------ neural models
class SeqModel(nn.Module):
    def __init__(self, n_items, kind, d=64, L=200, layers=2, heads=2, dropout=0.2):
        super().__init__()
        self.kind, self.L = kind, L
        self.emb = nn.Embedding(n_items + 1, d, padding_idx=0)          # item i -> index i + 1
        nn.init.normal_(self.emb.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        if kind == "sasrec":
            self.pos = nn.Embedding(L, d)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.ln = nn.LayerNorm(d)
        else:
            self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x):                                                # x: (B, T) item indices + 1, 0 = pad (left)
        h = self.emb(x)
        if self.kind == "sasrec":
            T = x.shape[1]
            h = self.drop(h + self.pos(torch.arange(T, device=x.device))[None])
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            h = self.ln(self.enc(h, mask=causal))                          # left padding: pads only precede real items
            h = h * (x > 0).unsqueeze(-1)
        else:
            h, _ = self.gru(self.drop(h))
        return h                                                         # (B, T, d)


def windows(seqs, L):
    """Non-overlapping windows of length L + 1 from the end of each sequence (inputs = w[:-1], targets = w[1:])."""
    out = []
    for s in seqs:
        for end in range(len(s), 1, -L):
            w = s[max(0, end - L - 1):end]
            if len(w) >= 2:
                out.append(w)
    return out


def pad_left(ws, L, dev):
    x = np.zeros((len(ws), L + 1), dtype=np.int64)
    for i, w in enumerate(ws):
        x[i, L + 1 - len(w):] = np.asarray(w) + 1
    return torch.as_tensor(x, device=dev)


@torch.no_grad()
def rank_positions(model, t, ev, L, dev, bs=256):
    model.eval()
    uid, item = t.uid.to_numpy(), t.item.to_numpy()
    E = model.emb.weight[1:]                                             # (n, d)
    pos = np.zeros(len(ev), dtype=np.int64)
    for i in range(0, len(ev), bs):
        b = ev[i:i + bs]
        hist = []
        for e in b:
            start = e - 1
            while start - 1 >= 0 and uid[start - 1] == uid[e] and e - start < L:
                start -= 1
            hist.append(item[start:e])
        x = pad_left(hist, L - 1, dev)                                   # (B, L) after padding to L
        h = model(x)[:, -1]
        s = h @ E.T                                                      # (B, n)
        tgt = torch.as_tensor(item[b], device=dev, dtype=torch.long)
        ts = s.gather(1, tgt[:, None])
        pos[i:i + len(b)] = (s > ts).sum(1).cpu().numpy()
    model.train()
    return pos


def run_neural(t, n, ev, kind, epochs, L=200, bs=128, K=2048, lr=1e-3, seed=0, test_days=1, return_model=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device()
    T_max = t.timestamp.max()
    val_mask = ((t.timestamp > T_max - (test_days + 1) * DAY) & (t.timestamp <= T_max - test_days * DAY)).to_numpy()
    train_df = t[t.timestamp <= T_max - (test_days + 1) * DAY]
    seqs = [g.to_numpy() for _, g in train_df.groupby("uid", sort=False)["item"]]
    ws = windows(seqs, L)
    cnt = np.bincount(train_df.item.to_numpy(), minlength=n).astype(np.float64)
    q = cnt ** 0.75; q /= q.sum()
    logq = torch.as_tensor(np.log(np.maximum(q, 1e-12)), device=dev, dtype=torch.float32)
    # negatives: a large pool drawn once on CPU (torch.multinomial over 600k+ classes exceeds MPS buffer limits)
    pool = torch.as_tensor(np.random.default_rng(seed).choice(n, size=20_000_000, p=q), device=dev)
    pool_i = 0
    val_ev = sample_events(val_mask, t.uid.to_numpy(), 3000, seed + 1)
    model = SeqModel(n, kind, L=L).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"{kind}: {len(ws):,} training windows, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, device {dev}")
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(len(ws)); t0 = time.time(); tot = 0.0
        for s in range(0, len(ws), bs):
            x = pad_left([ws[j] for j in perm[s:s + bs]], L, dev)
            inp, tgt = x[:, :-1], x[:, 1:]
            h = model(inp)
            m = tgt > 0
            h, tgt = h[m], tgt[m] - 1                                    # (P, d), item ids
            if pool_i + K > len(pool):
                pool_i = 0
            neg = pool[pool_i:pool_i + K]; pool_i += K
            E = model.emb.weight
            pos_logit = (h * E[tgt + 1]).sum(-1, keepdim=True) - logq[tgt][:, None]
            neg_logit = h @ E[neg + 1].T - logq[neg][None]
            neg_logit = neg_logit.masked_fill(neg[None] == tgt[:, None], float("-inf"))
            loss = F.cross_entropy(torch.cat([pos_logit, neg_logit], 1), torch.zeros(len(tgt), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        vpos = rank_positions(model, t, val_ev, L, dev)
        vhr = float((vpos < 10).mean())
        print(f"epoch {ep}: loss {tot / math.ceil(len(ws) / bs):.3f}, val HR@10 {vhr:.4f}, {time.time() - t0:.0f}s", flush=True)
        if vhr > best:
            best, best_state, bad = vhr, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    if return_model:
        return model, best
    return rank_positions(model, t, ev, L, dev), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sasrec", "gru4rec", "counts"])
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--max_windows", type=int, default=0, help="smoke test: cap training windows")
    a = ap.parse_args()
    t, n = load()
    uid, org, seen = t.uid.to_numpy(), t.is_organic.to_numpy(), t.seen.to_numpy()
    test_mask = (t.timestamp > t.timestamp.max() - DAY).to_numpy()
    ev = sample_events(test_mask, uid, a.n_test, 0)
    print(f"listen+ {len(t):,}; items {n:,}; test events {len(ev):,} (organic {org[ev].mean():.3f})")
    if a.model == "counts":
        res = {m: metrics(p, org[ev], seen[ev]) for m, p in run_counts(t, n, ev).items()}
    else:
        global windows
        if a.max_windows:
            _w = windows
            windows = lambda seqs, L: _w(seqs, L)[:a.max_windows]       # noqa: E731
        pos, best = run_neural(t, n, ev, a.model, a.epochs)
        res = {a.model: metrics(pos, org[ev], seen[ev])}
        res[a.model]["best_val_HR@10"] = best
    out = ROOT / "runs" / f"yambda50m_seq_pilot_{a.model}{'_smoke' if a.max_windows else ''}.json"
    out.write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        print(f"\n{m}")
        for g in GROUPS:
            x = r[g]
            print(f"  {g:<20} n={x['n']:>5}  HR@10 {x['HR@10']:.4f}  NDCG@10 {x['NDCG@10']:.4f}  HR@100 {x['HR@100']:.4f}")


if __name__ == "__main__":
    main()
