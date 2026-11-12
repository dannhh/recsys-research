# Yambda step 1: SASRec tuning vs count models (2026-11-13)

## Setup
- Script: `scripts/yambda_bench.py`, run on Kaggle T4 x2 by the kernel `dnnguynhng/yambda-bench-step1-tune` (built with `scripts/kaggle_build.py`).
- Data: Yambda-50M listens with played ratio >= 50% (29.4M events, 631k items).
- Split: the last 3 days are test days, with 10,000 sampled events per day (30,000 in total, fixed seed). Neural models train on events before the validation day, the day before the first test day. They early-stop on 10,000 validation events with patience 3 and a 100-minute budget.
- Count models (popularity, Markov-1, pairwise-5) ran on the Mac with the same test events, trained on everything before the first test day.
- Raw outputs: `runs/kaggle/yambda-bench-step1-tune/` (logs, per-run JSON, per-event ranks in NPZ) and `runs/yambda_bench/counts.*`.

## Results (pooled over 3 test days, HR@10, one seed)
| model | val HR@10 | all | organic repeat | organic new | rec. repeat | rec. new |
|---|---|---|---|---|---|---|
| Markov-1 | - | **0.132** | **0.294** | 0.153 | 0.018 | 0.007 |
| pairwise-5 | - | 0.128 | 0.278 | **0.157** | 0.022 | 0.008 |
| SASRec d64, lr 1e-3 | 0.098 | 0.109 | 0.234 | 0.079 | 0.029 | 0.014 |
| SASRec d128, lr 1e-3 | 0.113 | 0.131 | 0.265 | 0.095 | **0.056** | 0.015 |
| SASRec d256, lr 1e-3 | **0.114** | 0.127 | 0.273 | 0.119 | 0.028 | **0.020** |
| SASRec d128, lr 3e-4 | 0.092 | 0.104 | 0.229 | 0.077 | 0.024 | 0.012 |
| SASRec d256, lr 3e-4 | 0.081 | 0.075 | 0.156 | 0.062 | 0.024 | 0.013 |

HR@100 on all events: SASRec d256 0.329 and d128 0.326, against pairwise 0.259 and Markov 0.216.

- SASRec d64 with lr 3e-4 is not a result. Its validation HR@10 stayed near 0.006 while the loss was still falling, so early stopping cut it after 6 epochs. Patience 3 is too short for slow learning rates.
- SASRec d256 with lr 1e-3 ran 28 epochs at about 190 s each. Validation had flattened (0.114 at epoch 26, 0.113 at epoch 28), so it is close to converged.

## Reading
- **The ranking flip survives tuning (one seed).** The best SASRec is 4 to 5 times larger than the pilot model, and still loses to the count models on organic replay: 0.273 vs 0.294 on organic repeat, 0.119 vs 0.157 on organic new. It wins on recommender-driven events: 2.5x Markov on recommended-new, and 2.6x on recommended-repeat for d128.
- **The aggregate verdict still depends on the cutoff.** HR@10 on all events crowns Markov (0.132 vs 0.131 for d128, a near tie). HR@100 crowns SASRec by a wide margin.
- **Verdict: GO** for the next stage, pending seeds.
- Open question: d128 is far better than d256 on recommended-repeat (0.056 vs 0.028). This could be seed noise; check with seeds before reading anything into it.

## Next
1. Seeds 1 and 2 for SASRec d256 lr 1e-3 and d128 lr 1e-3 (4 runs, about 2 to 3 hours on T4 x2).
2. GRU4Rec and BERT4Rec at the best setting, 2 seeds each.
3. Paired bootstrap over test events (ranks are saved per event), per group and per day.
4. Replay-share decomposition (truncation and shuffling within each group), then the source-aware training method.
