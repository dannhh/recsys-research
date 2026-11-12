# recsys-research

Recommender-systems research toward a PhD application. The repo holds one project at a time; older directions stay in `notes/` and `scripts/`.

Start from `notes/status.md`; working rules are in `CLAUDE.md`, and `notes/README.md` indexes the notes.

## Current project
**What do next-event benchmarks measure?** Organic replay vs recommender-driven events on Yambda.
- Plan and evidence: `notes/plan-organic-vs-recommended.md`
- New-machine setup: `notes/ops/compute.md`
- Latest results: `notes/experiments/2026-11-13-yambda-step1-tune.md`

## Earlier directions
- Conformal structured sets over the Semantic-ID tree (stopped after the pilot): `notes/archive/design.md`, `notes/archive/2026-10-02-pilot-results.md`

## Layout
- `scripts/`: experiments, one file per study
- `notes/`: status, plan, experiments (CSV + dated notes), ops, paper references, archive
- `runs/`: small result files (JSON, per-event ranks); checkpoints are not tracked
- `kaggle/`: Kaggle kernel metadata built by `scripts/kaggle_build.py`
- `data/`: not tracked; see the setup steps in the plan
