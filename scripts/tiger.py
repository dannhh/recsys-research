"""TIGER-style generative recommender over Semantic IDs, with trie-constrained beam search.

- Encoder input: [user token] + SIDs of the last H items (H = 20), flattened.
- Decoder output: the 4 SID tokens of the next item.
- Decoding: custom level-by-level beam search. At each level the next-token distribution is
  renormalised over the trie children of the current prefix, so the path score of a prefix is
  its exact model probability under the trie-constrained distribution (needed for conformal sets).

Usage:
  python scripts/tiger.py train --data amazon2023_Musical_Instruments --epochs 60
  python scripts/tiger.py eval  --data amazon2023_Musical_Instruments --ckpt runs/<run>/best.pt
"""
import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
PAD, EOS, K, LEVELS, N_USER_TOK, HIST = 0, 1, 256, 4, 2000, 20
SID_OFFSET = 2
USER_OFFSET = SID_OFFSET + LEVELS * K
VOCAB = USER_OFFSET + N_USER_TOK


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, name):
        d = ROOT / "data" / "processed" / name
        self.sids = np.load(d / "sids.npy")                     # (n_items, 4)
        assert self.sids[:, 3].max() < K
        self.tok = self.sids + SID_OFFSET + np.arange(LEVELS) * K  # item -> 4 token ids
        self.loo = json.loads((d / "loo.json").read_text())
        self.n_items = len(self.sids)

    @staticmethod
    def user_tok(u):
        return USER_OFFSET + zlib.crc32(u.encode()) % N_USER_TOK

    def encode_hist(self, u, hist):
        h = hist[-HIST:]
        return [self.user_tok(u)] + self.tok[h].reshape(-1).tolist()

    def train_examples(self):
        X, Y = [], []
        for u, r in self.loo.items():
            s = r["train"]
            for j in range(1, len(s)):
                X.append(self.encode_hist(u, s[:j])); Y.append(s[j])
        return X, np.array(Y)

    def eval_examples(self, split):
        X, Y, U = [], [], []
        for u, r in self.loo.items():
            hist = r["train"] + ([r["valid"]] if split == "test" else [])
            X.append(self.encode_hist(u, hist)); Y.append(r[split]); U.append(u)
        return X, np.array(Y), U


def pad_batch(seqs, dev):
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    return ids.to(dev), (ids != PAD).to(dev)


# ---------------------------------------------------------------- trie
class Trie:
    """Level-indexed prefix tree over SID tuples. Node ids are per level; root is node 0 at level 0."""

    def __init__(self, sids):
        self.child = []           # child[l]: (n_nodes_l, K) int64, -1 if absent
        self.leaf_item = None     # (n_leaves,) item index
        nodes = {(): 0}
        level_nodes = [[()]]
        for l in range(LEVELS):
            nxt = {}
            for s in sids:
                p = tuple(s[:l + 1])
                if p not in nxt:
                    nxt[p] = len(nxt)
            ch = np.full((len(level_nodes[l]), K), -1, dtype=np.int64)
            pid = {p: i for i, p in enumerate(level_nodes[l])}
            for p, i in nxt.items():
                ch[pid[p[:-1]], p[-1]] = i
            self.child.append(ch)
            level_nodes.append(list(nxt.keys()))
        leaf_of = {p: i for i, p in enumerate(level_nodes[LEVELS])}
        self.leaf_item = np.empty(len(leaf_of), dtype=np.int64)
        for item, s in enumerate(sids):
            self.leaf_item[leaf_of[tuple(s)]] = item
        self.item_leaf = np.empty(len(sids), dtype=np.int64)
        self.item_leaf[self.leaf_item] = np.arange(len(self.leaf_item))
        self.child_t = None

    def to(self, dev):
        self.child_t = [torch.as_tensor(c, device=dev) for c in self.child]
        return self


# ---------------------------------------------------------------- model
def build_model():
    cfg = T5Config(vocab_size=VOCAB, d_model=128, d_kv=64, num_heads=6, d_ff=1024, num_layers=4,
                   num_decoder_layers=4, dropout_rate=0.1, feed_forward_proj="relu",
                   pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD)
    return T5ForConditionalGeneration(cfg)


@torch.no_grad()
def beam_search(model, trie, enc_ids, enc_mask, beam):
    """Returns leaf node ids (B, beam) and log-probs (B, beam), plus per-level prefix log-probs.

    Scores are sums of trie-renormalised log-probabilities, i.e. exact log p(prefix | x)
    under the constrained model. Fewer than `beam` valid paths are padded with -inf.
    """
    dev = enc_ids.device
    B = enc_ids.size(0)
    enc = model.encoder(input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state
    node = torch.zeros(B, 1, dtype=torch.long, device=dev)      # current node ids at level l
    score = torch.zeros(B, 1, device=dev)
    toks = torch.zeros(B, 1, 0, dtype=torch.long, device=dev)
    for l in range(LEVELS):
        nb = node.size(1)
        dec_in = torch.cat([torch.full((B, nb, 1), PAD, device=dev, dtype=torch.long), toks], dim=2).view(B * nb, -1)
        out = model.decoder(input_ids=dec_in, encoder_hidden_states=enc.repeat_interleave(nb, 0),
                            encoder_attention_mask=enc_mask.repeat_interleave(nb, 0)).last_hidden_state[:, -1]
        logits = model.lm_head(out * (model.model_dim ** -0.5))[:, SID_OFFSET + l * K: SID_OFFSET + (l + 1) * K]
        children = trie.child_t[l][node.clamp(min=0).view(-1)]            # (B*nb, K)
        valid = (children >= 0) & (node.view(-1, 1) >= 0)
        logp = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        logp = torch.nan_to_num(logp, nan=float("-inf"))
        cand = (score.view(-1, 1) + logp).view(B, nb * K)
        k = min(beam, nb * K)
        top, arg = cand.topk(k, dim=1)
        parent, code = arg // K, arg % K
        node = torch.gather(children.view(B, nb, K), 1, parent.unsqueeze(-1).expand(-1, -1, K))
        node = node.gather(2, code.unsqueeze(-1)).squeeze(-1)
        node = torch.where(torch.isfinite(top), node, torch.full_like(node, -1))
        toks = torch.cat([toks.gather(1, parent.unsqueeze(-1).expand(-1, -1, toks.size(2))),
                          (code + SID_OFFSET + l * K).unsqueeze(-1)], dim=2)
        score = top
    return node, score


def rank_metrics(pred_items, target, ks=(5, 10)):
    hit = pred_items == target[:, None]
    pos = np.where(hit.any(1), hit.argmax(1), 10**9)
    m = {}
    for k in ks:
        m[f"R@{k}"] = float((pos < k).mean())
        m[f"N@{k}"] = float(np.where(pos < k, 1 / np.log2(pos + 2), 0).mean())
    return m


def evaluate(model, trie, data, split, beam=50, bs=256, limit=None):
    model.eval(); dev = device()
    X, Y, _ = data.eval_examples(split)
    if limit:
        X, Y = X[:limit], Y[:limit]
    preds = []
    for i in range(0, len(X), bs):
        ids, mask = pad_batch(X[i:i + bs], dev)
        leaves, _ = beam_search(model, trie, ids, mask, beam)
        lv = leaves.cpu().numpy()
        preds.append(np.where(lv >= 0, trie.leaf_item[lv.clip(0)], -1))
    return rank_metrics(np.concatenate(preds), Y)


# ---------------------------------------------------------------- train
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = device(); data = Data(args.data); trie = Trie(data.sids).to(dev)
    model = build_model().to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M; device {dev}")
    X, Y = data.train_examples()
    print(f"train examples {len(X):,}")
    run = ROOT / "runs" / f"{args.data}_seed{args.seed}"; run.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(X) / args.bs); total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * max(0.05, 0.5 * (1 + math.cos(math.pi * s / total))))
    best, log = -1, []
    for ep in range(1, args.epochs + 1):
        model.train(); perm = np.random.permutation(len(X)); t0 = time.time(); tot = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.bs:(b + 1) * args.bs]
            ids, mask = pad_batch([X[i] for i in idx], dev)
            labels = torch.as_tensor(data.tok[Y[idx]], device=dev)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        msg = {"epoch": ep, "loss": tot / steps_per_epoch, "sec": round(time.time() - t0)}
        if ep % args.eval_every == 0 or ep == args.epochs:
            msg.update({f"val_{k}": v for k, v in evaluate(model, trie, data, "valid", beam=20, limit=5000).items()})
            if msg["val_N@10"] > best:
                best = msg["val_N@10"]; torch.save(model.state_dict(), run / "best.pt")
        if ep in args.save_at:
            torch.save(model.state_dict(), run / f"epoch{ep}.pt")
        log.append(msg); print(json.dumps(msg), flush=True)
        (run / "log.json").write_text(json.dumps(log, indent=1))


def evaluate_cmd(args):
    dev = device(); data = Data(args.data); trie = Trie(data.sids).to(dev)
    model = build_model().to(dev); model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    t0 = time.time()
    m = evaluate(model, trie, data, args.split, beam=args.beam)
    print(json.dumps({"split": args.split, "beam": args.beam, **m, "sec": round(time.time() - t0)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--data", default="amazon2023_Musical_Instruments")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--save_at", type=int, nargs="*", default=[5, 15, 30])
    ap.add_argument("--ckpt"); ap.add_argument("--split", default="test"); ap.add_argument("--beam", type=int, default=50)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate_cmd(a)
