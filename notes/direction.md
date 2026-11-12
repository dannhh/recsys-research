# Research direction and target labs (decided 2026-10-17)

Why this project was kept, how it should grow into a paper, and which labs it fits. Context for any future change of direction.

## Decision
- Keep organic vs recommended on Yambda, and grow it from a critique of benchmarks into **diagnosis plus a fix**. Working title: "Next-event benchmarks reward replay and policy imitation: diagnosis and source-aware training".
- Do not start LLM-based or multimodal recommendation from scratch. The big labs (He, Chua) publish dozens of papers a year there; one person with a few GPUs and 2 to 3 months is unlikely to stand out.
- The edge is the data: Yambda appears to be the only large public dataset with event-level organic vs recommended labels (dataset search in `plan-organic-vs-recommended.md`). Earlier work (Woolridge 2021, Klenitskiy 2026, Parajuli 2026) can only infer the mechanism; the label lets us test it.

## Paper plan
1. **Robustness (go/stop).** Tuned SASRec, GRU4Rec, BERT4Rec and count models; 3 seeds; 3 test days; metrics per event group. Go if model rankings still differ across groups after convergence. Stop if a tuned SASRec wins every group; the paper then becomes a measurement or reproducibility paper. Status 2026-11-13: GO on one seed.
2. **Mechanism.** Decompose each model's metric into replay, policy imitation and preference (truncation and shuffling within groups), with a simple mixture model of when mixed-log evaluation distorts rankings.
3. **Method.** Source-aware training: the event source as a conditioning token or a separate head, or IPS-style reweighting of recommender-driven events. Judge it on organic-new (discovery) events and on the fully observed KuaiRec matrix. A simple fix with a clear result is enough.
4. **Write-up.** arXiv preprint before PhD deadlines (usually December to January), then SIGIR full paper (deadline usually late January; check the exact date), RecSys as the fallback.
- Optional, if time allows: a small LLM-based recommender (about 0.5B, fine-tuned on item IDs) to ask whether LLM recommenders learn preference or imitate the logging policy. This links to the LLM-rec trend without competing head-on.

## Compute plan
- About 18 runs for the whole paper; one SASRec run takes 30 to 100 minutes on a T4. Kaggle T4 x2 runs two jobs at a time, so the paper needs a few sessions, a small share of the weekly quota.
- Main Kaggle account only. A second account breaks Kaggle's one-account rule and would risk the competition account.

## Target labs (claims checked 2026-10-17)
| Lab | Fit | Checked facts |
|---|---|---|
| Xiangnan He, USTC | best fit: debiasing, exposure, feedback loops; Chongming Gao (KuaiRec, KuaiRand, SPRec) is the person to read first | "Beyond Static Best-of-N: Bayesian List-wise Alignment for LLM-based Recommendation" on arXiv 2605.04559 (SIGIR 2026 acceptance not confirmed); "Reinforced Latent Reasoning for LLM-based Recommendation" at ICLR 2026, with He and Chua as authors |
| Tsinghua FIB Lab | good fit: causal recommendation, exposure and popularity bias | TOIS survey on causal inference in recommender systems exists; whether its repo is still maintained is unchecked |
| Tat-Seng Chua, NUS | fits the owner's LLaVA-Med thesis, not this project; works closely with He | the HM-RAG paper with Chua is the long-video journal article; the better-known ACM MM 2025 HM-RAG has no Chua; LLaVA-UHD v2 is mainly a Tsinghua paper |
| Jakob Runge | fits CarbonGlobe (climate, causal time series) better than recommendation | lab moved to the University of Potsdam in March 2025 (chair AI in the Sciences), branch at TU Berlin; the CausalEarth ERC project has ended; maintains Tigramite |

Sources for the lab checks: arxiv.org/abs/2605.04559, iclr.cc/virtual/2026/poster/10008315, arxiv.org/abs/2504.12330, pure.bit.edu.cn (HM-RAG long video), uni-potsdam.de/en/ais, tu-dresden.de (Runge appointment, 2024).
