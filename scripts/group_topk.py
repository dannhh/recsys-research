import sys; sys.path.insert(0, ".")
import numpy as np
from conformal import item_node_paths, quantile
from tiger import LEVELS, Data, Trie
from group_coverage import POP_BINS, bin_of
data = Data("amazon2023_Musical_Instruments"); trie = Trie(data.sids)
paths = item_node_paths(trie, data.sids)
X, Y, U = data.eval_examples("test")
cnt = np.zeros(data.n_items, dtype=np.int64)
for r in data.loo.values(): np.add.at(cnt, r["train"], 1)
item_g = bin_of(cnt, POP_BINS); leaf_g = item_g[trie.leaf_item]
perm = np.random.default_rng(0).permutation(len(X)); rest = perm[len(X)//2:]
exp_users = rest[:2000]; leaf = np.load("../runs/amazon2023_Musical_Instruments_seed0/best_leaf_2000.npy")
tl = paths[Y[exp_users], LEVELS-1]; ts = leaf[np.arange(2000), tl]; tg = item_g[Y[exp_users]]
# rank of target within its own popularity group
wr = np.array([(leaf[n][leaf_g == tg[n]] > ts[n]).sum() + 1 for n in range(2000)])
print("group sizes (items):", [int((leaf_g == g).sum()) for g in range(4)])
for al in (0.7, 0.5, 0.3):
    ks = [int(quantile(wr[:1000][tg[:1000] == g].astype(float), al)) for g in range(4)]
    cov = wr[1000:] <= np.array(ks)[tg[1000:]]
    print(f"target {1-al:.1f} k_g {ks} size {sum(ks)} cov {cov.mean():.3f} |",
          " ".join(f"{POP_BINS[g][2]} {cov[tg[1000:]==g].mean():.2f}" for g in range(4)))
