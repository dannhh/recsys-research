# Status

Updated 2026-11-13 (+07). Read this first after a break or a compaction; details live in the linked notes. Working rules are in `CLAUDE.md` at the repo root.

| | |
|---|---|
| Project | What do next-event benchmarks measure? Organic replay vs recommender-driven events (`plan-organic-vs-recommended.md`) |
| Target | RecSys / SIGIR full paper; arXiv preprint before PhD deadlines |
| Latest result | Step 1 GO on one seed: a tuned SASRec (d256) still loses to count models on organic replay and wins on recommender-driven events (`experiments/2026-11-13-yambda-step1-tune.md`) |
| Compute | Kaggle T4 x2 on `dnnguynhng`; about 190 s per epoch for SASRec d256, 30 to 100 minutes per run |

## Where we are
- The predictability gap between event sources holds on 3 datasets (Yambda, KuaiSAR, KuaiRand).
- A model-ranking change across event groups is shown on Yambda only. KuaiRec is weak evidence: its reversal comes from the popular head.
- Step 1 tuning (6 SASRec configs, 1 seed) keeps the flip. The best configs are d256 and d128 with lr 1e-3.

## Next steps
1. **Running (2026-11-13):** kernel `dannhcse/yambda-bench-step1-scale`: SASRec d512 seeds 0-1, d256 with 4 layers seed 0, d256 seeds 1-2 (lr 1e-3, 150-minute budget each, about 5 hours). It tests the main risk: in step 1 the gap to the count models on organic events shrank as d grew (organic new 0.079, 0.095, 0.119 for d64, d128, d256 vs 0.157), so a larger model might erase the flip.
2. GRU4Rec and BERT4Rec at the best setting, 2 seeds each.
3. Paired bootstrap over test events, per group and per day, from the saved NPZ ranks.
4. Replay-share decomposition, then the source-aware training method (plan, steps 3 to 5).

## Open risks
- Generality: the flip rests on Yambda alone.
- Scoop: Yandex lists organic vs algorithmic analysis as its own future work.
