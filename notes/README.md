# Notes

| File | What it holds |
|---|---|
| `status.md` | current project, latest result, next steps (read first; working rules in `../CLAUDE.md`) |
| `plan-organic-vs-recommended.md` | the current project: claim, evidence across datasets, prior work, steps, risks |
| `experiments/experiments.csv` | one row per run: dataset, model, config, seed, per-group metrics, source file |
| `experiments/2026-10-03-yambda-organic-pilot.md` | the first Yambda pilot and the prior-work reading |
| `experiments/2026-11-13-yambda-step1-tune.md` | step 1: tuned SASRec vs count models on 3 test days |
| `ops/compute.md` | machines, Kaggle kernels, data download, timings |
| `paper/references.md` | every cited source, with where and when it was checked |
| `archive/` | the stopped conformal-GR direction (design, data checks, pilot, PCTM probe) |

## Naming
- Dated notes: `YYYY-MM-DD-<dataset>-<study>.md`.
- Run tags are built by `scripts/yambda_bench.py` from the config (`<model>_d<d>_l<layers>_lr<lr>_s<seed>`). Never name a run by hand.
- Kaggle kernel slugs: `<dataset>-<study>-<step>`, for example `yambda-bench-step1-tune`.
