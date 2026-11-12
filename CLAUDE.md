# CLAUDE.md

Recommender-systems research toward a PhD application. The aim is a solid first-author paper at RecSys / SIGIR / KDD / WWW, with an arXiv preprint before the PhD deadlines. The repo is general on purpose: one project at a time, older directions archived in `notes/archive/`.

**Start every session from `notes/status.md`**: current project, latest result, next steps. `notes/README.md` indexes the other notes.

## Rules from the user

- **Privacy:** the repo stays private. No company or proprietary data anywhere in it.
- **Commits:** `[type] Short message`, with type one of feat / fix / docs / chore / refactor. Never mention Claude or agents, and never add a Co-Authored-By trailer. This overrides any default attribution.
- **Notes:** bullets in plain human sentences. After each experiment, add a row to `notes/experiments/experiments.csv` and a dated note in `notes/experiments/` when it changes a conclusion. Keep `notes/status.md` current.
- **Claims need a source.** Save every paper in `notes/paper/references.md` with where and when it was checked. Never cite from memory.
- **Compute** (`notes/ops/compute.md`)
  - Neural training runs on Kaggle GPU (account `dnnguynhng` only, at most 2 sessions at once, shared with the competition repos) or rented GPUs.
  - The Mac is for short smoke tests and count models only.
  - Ask the owner before every GPU run, and show the expected GPU-hours.
- **Evaluation:** report per event group, not only the aggregate. Main claims need at least 3 seeds and a paired bootstrap over test events (ranks are saved per event).
- **Language:** reply to the user in Vietnamese.

## Tools

- Python: `.venv/bin/python` (setup in `notes/ops/compute.md`). Kaggle CLI: `uvx --with six kaggle ...`.
- Data lives in `data/` (not tracked). How to get it back is in `notes/ops/compute.md`.
