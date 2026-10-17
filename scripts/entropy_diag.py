"""Does the model know which users are hard? (pilot, step 4)

From the cached full-tree leaf scores (group_coverage.py): per-user entropy of p(item | x), its link to the
target rank, and whether any score can make sets adaptive. Compared at the same target coverage:
  global top-k      one k for everyone
  entropy-bin top-k k per entropy quintile (oracle-ish upper bound on what an entropy-aware set can gain)
  APS               smallest top set whose probability mass reaches a calibrated level
Calibration on the first 1,000 expanded users, evaluation on the other 1,000.
Usage: python scripts/entropy_diag.py
"""
import numpy as np
from scipy.stats import spearmanr

from conformal import item_node_paths, quantile
from tiger import LEVELS, Data, Trie

RUN = "../runs/amazon2023_Musical_Instruments_seed0/"
data = Data("amazon2023_Musical_Instruments"); trie = Trie(data.sids)
paths = item_node_paths(trie, data.sids)
X, Y, U = data.eval_examples("test")
perm = np.random.default_rng(0).permutation(len(X)); exp_users = perm[len(X) // 2:][:2000]
lp = np.load(RUN + "best_leaf_2000.npy").astype(np.float64)
p = np.exp(lp)
p /= p.sum(1, keepdims=True)
tl = paths[Y[exp_users], LEVELS - 1]
pt = p[np.arange(len(p)), tl]
rank = (p > pt[:, None]).sum(1) + 1
H = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
top1 = p.max(1)
order = np.argsort(-p, 1)
cum = np.cumsum(np.take_along_axis(p, order, 1), 1)

C, T = slice(0, 1000), slice(1000, 2000)
print(f"entropy nats: p10 {np.percentile(H, 10):.2f} median {np.median(H):.2f} p90 {np.percentile(H, 90):.2f}"
      f"  (uniform over {p.shape[1]} items = {np.log(p.shape[1]):.2f}); perplexity median {np.exp(np.median(H)):.0f}")
print(f"top-1 prob: p10 {np.percentile(top1, 10):.4f} median {np.median(top1):.4f} p90 {np.percentile(top1, 90):.4f}")
print(f"Spearman(entropy, log rank) {spearmanr(H, np.log(rank))[0]:.3f}; Spearman(top1, log rank) {spearmanr(top1, np.log(rank))[0]:.3f}")

edges = np.quantile(H[C], [0.2, 0.4, 0.6, 0.8]); b = np.digitize(H, edges)
print("\nentropy quintile: H range | median target rank | hit@1000 | perplexity")
for g in range(5):
    m = b == g
    print(f"  Q{g + 1}: {H[m].min():.2f}-{H[m].max():.2f} | {np.median(rank[m]):6.0f} | {(rank[m] <= 1000).mean():.2f} | {np.exp(np.median(H[m])):.0f}")

aps_score = cum[np.arange(len(p)), (order == tl[:, None]).argmax(1)]  # mass up to and including target
print("\ntarget | global top-k: size cov | entropy-bin top-k: mean size cov, k per quintile | APS: mean size, p10-p90, cov")
for al in (0.7, 0.5, 0.3):
    k = quantile(rank[C].astype(float), al)
    kb = np.array([quantile(rank[C][b[C] == g].astype(float), al) for g in range(5)])
    tau = quantile(aps_score[C], al)
    aps_size = (cum[T] < tau).sum(1) + 1
    print(f"  {1 - al:.1f} | {k:.0f} {(rank[T] <= k).mean():.3f} | {kb[b[T]].mean():.0f} {(rank[T] <= kb[b[T]]).mean():.3f} {kb.astype(int).tolist()}"
          f" | {aps_size.mean():.0f} {np.percentile(aps_size, [10, 90]).astype(int).tolist()} {(aps_size >= rank[T]).mean():.3f}")
