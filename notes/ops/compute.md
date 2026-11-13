# Compute

## New machine setup
- `python -m venv .venv && .venv/bin/pip install -r requirements.txt` (the pinned versions were used on Python 3.9).
- Yambda listens: `uvx --with six kaggle datasets download dnnguynhng/yambda-50m-listens -p data/raw/yambda --unzip`. This is a private copy of `listens_50m.parquet` from the Yambda release (Apache-2.0).
- The other datasets (KuaiRand, KuaiRec, KuaiSAR, Amazon 2023, FINN.no) must be downloaded again from their sources. The links are in `../plan-organic-vs-recommended.md`. Each pilot script reads from `data/raw/<name>/`.

## Mac (M4 Pro, 24 GB)
- Count models on Yambda-50M (3 test days, 30,000 events): under 1 minute.
- Peak RAM when loading Yambda-50M and building training windows: about 5 GB.
- Use it for smoke tests only (`--max_windows 2000 --epochs 1`).

## Kaggle (T4 x2)
- Accounts: `dnnguynhng` (shared with the competition repos) and, from 2026-11-13 by the owner's choice, `dannhcse` for this project. The token is passed as `KAGGLE_API_TOKEN` in the shell and never written to the repo. Each account holds its own private copy of `yambda-50m-listens`; pass `--user <account>` to `scripts/kaggle_build.py`.
- The runner stops at once if the session has no GPU, instead of training on CPU.
- 2026-11-13: the first kernel on `dannhcse` got no GPU (no `nvidia-smi` in the session) and failed within a minute. Kaggle gives GPUs only to phone-verified accounts; verify the phone in that account's settings before pushing GPU kernels there.
- Build a kernel: `.venv/bin/python scripts/kaggle_build.py --slug <slug> --jobs "<args for yambda_bench.py>" ...`
- Push: `uvx --with six kaggle kernels push -p kaggle/<slug>`. Download: `uvx --with six kaggle kernels output dnnguynhng/<slug> -p runs/kaggle/<slug>`.
- The runner gives each GPU its own job queue worker, so T4 x2 runs two jobs at a time.
- Timings: SASRec d64 about 70 s per epoch, d256 about 190 s per epoch. One step-1 kernel (6 runs, 6 hours of run time in total) took about 3 to 4 hours of wall time.
- The account is shared with the competition repos: at most 2 sessions at once, about 30 GPU hours per week. A kernel can sit in QUEUED for a while at busy hours.
- Early stopping uses patience 3. That is too short for lr 3e-4 at d64, which stalled and was cut after 6 epochs.
