# Conformal prediction sets for Semantic-ID generative recommendation — design

Status: draft v1, 2026-10-01. Merges an external design proposal (SID + trie + per-level conformal decoding + vLLM serving) with the literature check in the vault note "Open Problems in Generative Recommendation". Claims marked *(hypothesis)* are what the pilot must test.

## 1. Problem

Semantic-ID (SID) generative recommenders (TIGER-style) return a top-K list from beam search with no statement of how likely the item the user actually wants is in it. Two practical failures follow:

1. **No guarantee.** A fixed K is too long for easy users and too short for hard ones.
2. **New items are unreachable.** Under a temporal split, SID generators almost never produce the full SID of an item unseen in training, although they often predict its first tokens correctly (cold-item reachability study, 2026-07).

Not a contribution: invalid-ID "hallucination". Trie-constrained decoding already removes it (NEO; TIGER without a trie has 0.1–1.6% invalid IDs). We use the trie as standard infrastructure.

## 2. Idea

Return a **structured prediction set**: a small set of nodes of the SID prefix tree, mixing exact items (leaves) where the model is confident with coarse prefixes (a region of the catalogue) where it is not, with a finite-sample guarantee that the target item lies under one of the returned nodes with probability ≥ 1 − α.

GR-specific property that makes this cheap: for an autoregressive SID model with a trie-normalised softmax, the probability of a prefix node is **exact and available without enumerating its leaves**:

  p(prefix c₁..c_d | x) = ∏_{l ≤ d} p(c_l | c_<l, x)

Flat models (SASRec, two-tower) must sum over all descendant items to get the same quantity.

## 3. Planned contributions

- **C1 Method.** Conformal structured sets over the SID tree for generative recommenders, using exact prefix marginals and beam-limited search, with a coverage guarantee for the *final item*.
- **C2 Tokenizer analysis.** How set efficiency depends on the tree: learned SID tree (RQ-VAE / Residual K-means) vs the human category tree vs a random tree; effect of collisions and boundary items. *(hypothesis: a learned tree gives smaller sets than a random tree and is competitive with the human tree)*
- **C3 New items and drift.** Node-level coverage for items unseen at training time (new items get content SIDs at arrival and are inserted into the trie without retraining), and coverage over time with adaptive calibration. *(hypothesis: item-level sets lose coverage on new items; node-level sets keep it with moderate size)*
- **C4 Protocol.** A reusable evaluation protocol for conformal generative recommendation (splits, groups, metrics, baselines).

## 4. Method

### 4.1 Base model
TIGER-style encoder–decoder over SID tokens (L = 3 semantic levels × K = 256 codes, plus a disambiguation token where needed), trained with next-item cross-entropy. Trie-constrained decoding. Item SIDs from content embeddings (text for Amazon, audio for Yambda) via Residual K-means; RQ-VAE as an ablation.

### 4.2 Scores and calibration (the fix to the external proposal)
The external proposal calibrated a separate threshold q̂_l per level and pruned level by level. That does **not** guarantee coverage of the final item: errors compound across levels (three levels at 90% each can give ≈ 73%). We calibrate on a **single path-level score** instead:

- Leaf score: s(x, y) = −log p(y | x) = −Σ_l log p(y_l | y_<l, x), from the trie-constrained model.
- Item-level set: C_item(x) = { leaves i : −log p(i | x) ≤ q̂ }, with q̂ the ⌈(n+1)(1−α)⌉/n empirical quantile of calibration scores (split conformal). Valid under exchangeability of calibration and test pairs.
- **Beam limitation.** Leaves outside the explored beam get score +∞. Then coverage is still valid but conservative, and capped by beam recall (method 1 of conformal beam search, AAAI 2024). The adaptive-beam variant (their method 2: keep expanding while the path score is under the threshold) removes the cap and is the default.
- Alternative kept for comparison: APS-style randomised scores over leaves found by search.

### 4.3 Structured (mixed-level) sets
For a threshold τ, build S_τ(x) top-down: start from the root; a node is **returned** if its own mass is ≥ τ-dependent cutoff and expanding it would not reduce the covered-item cost enough; otherwise expand its children. Requirements:

- **Nested in τ** (S_τ covers more as τ grows), so a single τ̂ can be chosen by split conformal on s(x, y) = inf{τ : y ∈ desc(S_τ(x))}.
- If a useful construction is not nested, fall back to learn-then-test calibration as in Conformal Structured Prediction (ICLR 2025).
- Cost of a set = number of items covered (primary) and number of nodes (readability). Depth of returned nodes = specificity.

### 4.4 Drift
Calibration data come from the past, test data from the future, so exchangeability fails. Options, compared:
- static split conformal (baseline, expected to drift);
- recalibration on a sliding window of the most recent period;
- adaptive conformal inference (Gibbs & Candès 2021) updating α_t online from realised coverage.

### 4.5 Serving (out of scope for the core study)
Branching decode shares the prompt's KV cache across branches; vLLM already does this for beam search. Prefix caching mainly saves **decode** compute across branches, not time-to-first-token (prefill of the user history dominates TTFT). Report decode cost (forward passes, beam width) only. A vLLM integration can be an appendix.

## 5. Data and splits

| Dataset | Role | Notes (from data checks 2026-10-01) |
|---|---|---|
| Amazon Reviews 2014 Beauty (5-core) | comparability with GR papers; **learned vs human tree** | 12,101 items, 100% with category paths (8 / 42 / 159 nodes at depths 2 / 3 / 4); only 517 unseen targets under a temporal split |
| Yambda-50M (listens) | **new items, drift, user-activity groups, deeper tree** | 877K items, 301 days, audio embeddings for 93.4% of items; 811 unseen test targets in the last week |
| Amazon 2023 large category or Instacart (optional) | second domain with a human hierarchy at scale | not checked yet |

Splits: leave-one-out on Beauty for comparability; global temporal split for all claims about new items and drift: train → validation (checkpoint selection) → calibration → test, each a later time window; calibration data unseen by the model.

## 6. Baselines

1. Top-K beam list, no guarantee (K tuned to match average set size).
2. Split conformal on item scores (leaf sets only).
3. Conformal beam search (AAAI 2024), both methods.
4. Conformal structured sets on: learned SID tree / human category tree / random tree of the same shape.
5. Same pipeline on an ID-based model (SASRec; leaf probabilities by full softmax, prefix mass by summation) to isolate what SIDs add.
6. Drift: static vs windowed recalibration vs adaptive conformal inference.
7. Optional: CPFT-style conformal-aware training; two-stage risk control.

## 7. Metrics

**Headline (what the main table compares across methods):**
1. **Average set size in items covered, at target coverage 1 − α = 0.9** (also 0.8), among methods whose realised coverage meets the target. A returned prefix counts as all items under it, so mixed-level and item-only sets are comparable. Plus the full size–coverage curve over α.
2. **Worst-group coverage**, especially for unseen (new) items: the result most likely to separate methods, since GR top-K lists score near zero there.
3. **Bridge to GR leaderboards:** at the same average budget (top-K with K = mean conformal set size), Recall@K of the fixed list vs coverage of the adaptive set.

**Full list:**
- Marginal coverage (must be ≥ 1 − α within sampling error).
- Conditional coverage by: seen vs unseen target, target popularity (unseen / tail / mid / head), user history length; report worst-group coverage.
- Set cost: items covered, nodes returned; specificity (mean depth).
- Utility at matched cost: hit rate of a top-K list with K = the conformal set's item count (shows what the guarantee costs or buys).
- Coverage over time (per week on Yambda).
- Decode cost: forward passes per user, beam-width distribution (median, p99), wall-clock p50 / p99 against a fixed beam of 30–50 (production-typical; GLIDE uses 30); size–cost curve at fixed coverage. Calibration is offline and the online threshold check is negligible; the cost risk is search depth for uncertain users. *(hypothesis: mixed-level sets need fewer decode steps than item-only sets at the same coverage, because they stop at a prefix where item sets must descend to leaves)*

## 8. Pilot (go / no-go), on this Mac

1. Amazon Beauty: Residual K-means SIDs → TIGER → reproduce Recall@10 ≈ 0.065 (sanity check) → item sets vs structured sets vs conformal beam search at α ∈ {0.1, 0.2} → learned vs human tree. Save checkpoints at several training stages and evaluate each, to see whether the prefix-set advantage shrinks as the model gets stronger.
2. Yambda-50M (subsampled: ~100 targets per user, last 50 listens as context; measured ~26 min per epoch on MPS): audio SIDs → TIGER → seen / unseen coverage, weekly coverage.

Go if: coverage holds, structured sets are clearly smaller or more specific than item sets at the same coverage, the advantage does not collapse at later checkpoints, and unseen-item coverage is materially better with node-level sets. Stop or rethink otherwise.

## 9. Risks

- Reads as an application of known conformal tools (conformal beam search + structured prediction). Mitigation: C2 and C3 are GR-specific; lead with them.
- On small catalogues the SID tree is shallow (Beauty: ~47 items per first-level code, near-singletons below), so structured sets may show little benefit there. Mitigation: Yambda.
- Stronger base models shrink item sets, possibly erasing the gain. Mitigation: checkpoint-strength analysis in the pilot; full-scale runs on a GPU before any claim.
- "Conformal is slow": adaptive beams can blow up tail latency for uncertain users. Mitigation: beam caps (coverage stays valid but conservative; report the effect), stopping at prefixes, KV sharing across branches, and the precomputed-candidate setting common in production; measure p99, not only averages.
- Fast-moving area (UCSD / Snap, Meta, Alibaba groups active on adjacent questions): post a preprint as soon as results are clean.

## 10. References (see vault source notes)
Conformal beam search (AAAI 2024); Conformal Structured Prediction (ICLR 2025); Hierarchical Conformal Classification (2025); Conformal Language Modeling (ICLR 2024); missing-mass conformal (2025); CPFT (2024); FDR-controlled recommendation (Angelopoulos et al. 2022); conformal risk control for unwanted recommendations (RecSys 2025); two-stage risk control (2024); adaptive conformal inference (Gibbs & Candès 2021); TIGER (NeurIPS 2023); NEO / GLIDE (Spotify, 2026); cold-item reachability (2026-07); Time to Split (RecSys 2025); Yambda (2025).
