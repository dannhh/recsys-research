# Pilot results, Amazon 2023 Instruments (2026-10-02)

Backbone: TIGER-style T5 (4.8M params), Residual K-means SIDs (3x256 + disambiguation), 40 epochs, best = epoch 30.
Runs: `runs/amazon2023_Musical_Instruments_seed0/` (`best_test_eval.json`, `best_curve.json`, `epoch5_conformal_alpha0.1_partial.txt`).

## Backbone on test (leave-one-out, full catalog, beam 50)
R@5 0.0314, N@5 0.0200, R@10 0.0501, N@10 0.0260.
Published TIGER on this split: R@10 0.0537 (EPIC table) to 0.0566 (SPRINT, Latte tables). Ours is 7-11% below.

## Conformal sets (best.pt; calibration on 28,719 test users, sizes on 300 held-out users)
Coverage over all 28,720 held-out users (by score) matches 1 - alpha at every depth (e.g. 0.896 at alpha 0.1, 0.497 at 0.5). Validity holds.

| target coverage | item set, mean items | depth-1 prefix set, mean nodes (items under them) | depth-2, mean nodes |
|---|---|---|---|
| 0.1 | 31 | 1.7 (median 165 items; 32% of sets empty) | 16 |
| 0.2 | 126 | 5.2 (663) | 59 |
| 0.3 | 316 | 10 (1,282) | 140 |
| 0.5 | 1,247 | 25 (3,190) | 512 |
| 0.7 | 3,783 | 52 (6,418) | 1,492 |
| 0.9 | 12,011 | 118 (12,963) | 4,630 |

Top-k recall of the same model: k=10 0.053, 100 0.213, 1000 0.487, 2000 0.630, 5000 0.767, 10000 0.860. Median target rank 1,092.

## Findings
1. **90% coverage is out of reach with small sets.** It needs about half the catalog. Next-item prediction is intrinsically uncertain (R@10 about 5%), so useful alphas are 0.5-0.9.
2. **Item threshold sets are about as large as top-k at the matched recall** (50%: 1,247 items vs top-1000 at 48.7%). Set sizes barely vary across users (median about equal to mean), so the model's probabilities do not adapt the set size per user. Adaptivity, the usual selling point of conformal sets, is absent with this backbone.
3. **Prefix sets return far fewer nodes** (50% coverage: 25 depth-1 nodes vs 1,247 items, 50x fewer) but cover more items (3,190). Same trade-off as predicted in the design.
4. **Cost:** full tree expansion is 1.18 s per user on the M4 Pro (MPS); the alpha 0.1 threshold search on epoch 5 took about 16 s per user under contention.

## Open questions this raises
- Does a stronger backbone (published-strength TIGER, or SPRINT at item level) make set sizes adaptive? If not, conformal adds a guarantee but no efficiency over calibrated top-k.
- Framing: recall-guaranteed candidate generation (coverage = recall of the retrieval stage) fits the 0.3-0.7 coverage regime better than "90% guaranteed recommendation".
- Conditional coverage (unseen items, long-tail users) is still untested and may be where the method earns its place.

## Group-conditional coverage (`scripts/group_coverage.py`, `scripts/group_topk.py`)
2,000 held-out users expanded over the full tree: 1,000 calibrate top-k ranks, 1,000 are test users. Conformal thresholds are calibrated on the 28,719 cal users. Output: `best_groups.json`, cache `best_leaf_2000.npy`.
Groups:
- Item popularity, by the target's train count: tail<=5 (13% of test users, 8,000 items), 6-25 (36%), 26-100 (29%), head>100 (22%, 453 items).
- User history length: <=4 (26%), 5-9 (50%), >=10 (24%).

Coverage per item-popularity group, at target coverage 0.3 / 0.5 / 0.7:

| method | mean size | tail<=5 | 6-25 | 26-100 | head>100 |
|---|---|---|---|---|---|
| top-k (k 336 / 1,325 / 4,181) | 336 / 1,325 / 4,181 | 0.02 / 0.10 / 0.23 | 0.10 / 0.30 / 0.60 | 0.29 / 0.68 / 0.93 | 0.79 / 0.95 / 1.00 |
| marginal conformal | 313 / 1,253 / 3,848 | 0.02 / 0.08 / 0.21 | 0.11 / 0.30 / 0.58 | 0.28 / 0.66 / 0.91 | 0.77 / 0.95 / 1.00 |
| Mondrian by user activity | 322 / 1,262 / 3,862 | same as marginal | | | |
| class-conditional by item popularity | 1,082 / 3,227 / 7,324 | 0.31 / 0.53 / 0.69 | 0.33 / 0.53 / 0.73 | 0.28 / 0.46 / 0.71 | 0.28 / 0.46 / 0.67 |
| top-k per popularity group | 1,237 / 2,923 / 7,066 | 0.32 / 0.48 / 0.71 | 0.32 / 0.50 / 0.72 | 0.27 / 0.41 / 0.66 | 0.27 / 0.45 / 0.68 |

5. **The failure is along item popularity, not user history.**
   - Top-k and marginal conformal almost never cover tail targets: 2% at target 0.3, 23% at 0.7.
   - Head targets are over-covered.
   - Coverage across history-length groups is already flat for top-k (0.30 / 0.29 / 0.31 at target 0.3), so Mondrian by user activity fixes nothing.
6. **Class-conditional conformal restores per-group coverage.** The cost is 3.5x / 2.6x / 1.9x the set size, and it moves coverage from head items to tail items rather than adding recall.
7. **Per-group top-k does the same at about the same size** (1,237 vs 1,082; 2,923 vs 3,227; 7,066 vs 7,324). With this backbone, the group-conditional result comes from the grouping, not from conformal scores. The conformal version keeps one advantage: a finite-sample guarantee per group, calibrated on many more users (about 3,800 tail users vs 133 for per-group top-k).

## Does the model know which users are hard? (`scripts/entropy_diag.py`)
Uses the same 2,000 cached users: calibration on 1,000, evaluation on 1,000.
- Per-user entropy of p(item | x) is 7.65-8.59 nats (p10-p90); uniform over 24,587 items would be 10.11. Perplexity runs from about 2,100 (lowest quintile) to 5,350 (highest).
- The signal is weak but real:
  - Spearman(entropy, log target rank) = 0.20.
  - Median target rank by entropy quintile: 502 / 922 / 1,498 / 1,630 / 1,886.
  - hit@1000 by entropy quintile: 0.61 to 0.38.

| target coverage | global top-k, size | top-k per entropy quintile, mean size (k range) | APS, mean size (p10-p90) |
|---|---|---|---|
| 0.3 | 336 | 459 (94-736) | 388 (185-609) |
| 0.5 | 1,325 | 1,467 (424-1,887) | 1,418 (737-2,103) |
| 0.7 | 4,181 | 4,137 (1,444-5,399) | 3,952 (2,303-5,536) |

8. **Adaptive sets are possible but never smaller on average.**
   - Entropy-aware sets and APS vary 3-8x across users, so they do adapt.
   - Their mean size matches or exceeds global top-k at every coverage level.
   - This is the expected result. A threshold on p(item | x) already minimises expected set size for marginal coverage (Sadinle et al. 2019), and top-k is close to that threshold here.
   - Adaptivity can only buy conditional coverage across difficulty levels, never a smaller average set.
   - A stronger backbone helps only if it makes p(item | x) much more uneven across users. With SOTA R@10 at 0.07, that is unlikely on this dataset.
   - Verdict: the "smaller or adaptive sets than top-k" claim is not viable here, whatever the backbone.
