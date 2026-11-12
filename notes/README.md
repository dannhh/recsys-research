# Notes

| File | What it holds |
|---|---|
| `status.md` | current project, latest result, next steps (read first; working rules in `../CLAUDE.md`) |
| `direction.md` | why this project, the paper plan (diagnosis + fix), compute plan, target labs with checked facts |
| `plan-organic-vs-recommended.md` | the current project: claim, evidence across datasets, prior work, steps, risks |
| `experiments/experiments.csv` | one row per run: dataset, model, config, seed, per-group metrics, source file |
| `experiments/2026-10-03-yambda-organic-pilot.md` | the first Yambda pilot and the prior-work reading |
| `experiments/2026-11-13-yambda-step1-tune.md` | step 1: tuned SASRec vs count models on 3 test days |
| `ops/compute.md` | machines, Kaggle kernels, data download, timings |
| `paper/references.md` | every cited source, with where and when it was checked |
| `archive/` | the stopped conformal-GR direction (design, data checks, pilot, PCTM probe) |

## Outside the repo
- Obsidian vault (synced with Obsidian Sync): `wiki/projects/RecSys Research.md` (context, decisions, timeline), `wiki/concepts/Open Problems in Generative Recommendation.md` (literature map and the stopped directions), `wiki/concepts/Decision Models for Reranking.md`, `wiki/entities/Yambda.md`.
- Kaggle: private dataset `dnnguynhng/yambda-50m-listens`, kernels `dnnguynhng/yambda-bench-*`.
- Not kept anywhere: conformal-GR checkpoints in `runs/amazon2023_Musical_Instruments_seed0/` (local only, archive direction). `data/processed/` is rebuilt by `scripts/check_yambda.py` and the `prepare_*.py` scripts.

## Naming
- Dated notes: `YYYY-MM-DD-<dataset>-<study>.md`.
- Run tags are built by `scripts/yambda_bench.py` from the config (`<model>_d<d>_l<layers>_lr<lr>_s<seed>`). Never name a run by hand.
- Kaggle kernel slugs: `<dataset>-<study>-<step>`, for example `yambda-bench-step1-tune`.
