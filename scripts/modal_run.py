"""Run yambda_bench.py jobs on Modal GPUs, one container per job, all in parallel.

Setup (once):
  uvx modal token new
  uvx modal volume create yambda-data
  uvx modal volume put yambda-data data/raw/yambda/listens_50m.parquet /listens_50m.parquet
Run:
  uvx modal run --detach scripts/modal_run.py --jobs "--model sasrec --d 512 --seed 0; --model sasrec --d 256 --seed 1"
  uvx modal volume get recsys-runs /yambda_bench runs/modal/      # results (JSON, NPZ, logs)
Set MODAL_GPU (T4, L4, A10G, ...) to change the GPU type; the default is L4. Results land in the `recsys-runs` volume, so a detached run
keeps them even if this terminal disconnects.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("recsys-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy==2.0.2", "pandas==2.3.3", "pyarrow==21.0.0", "scipy==1.13.1")
    .add_local_file(HERE / "yambda_seq_pilot.py", "/code/yambda_seq_pilot.py")
    .add_local_file(HERE / "yambda_bench.py", "/code/yambda_bench.py")
)
data = modal.Volume.from_name("yambda-data")
runs = modal.Volume.from_name("recsys-runs", create_if_missing=True)
GPU = os.environ.get("MODAL_GPU", "L4")


@app.function(image=image, gpu=GPU, memory=16384, timeout=5 * 3600, volumes={"/data": data, "/runs": runs})
def run_bench(job: str) -> int:
    name = job.replace("--", "").replace(" ", "_")
    log = Path("/runs/yambda_bench/logs") / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, YAMBDA_DIR="/data", RUNS_DIR="/runs", PYTHONUNBUFFERED="1")
    with open(log, "w") as f:
        subprocess.run(["nvidia-smi"], stdout=f, stderr=subprocess.STDOUT, check=False)
        f.flush()
        r = subprocess.run([sys.executable, "yambda_bench.py", *shlex.split(job)], cwd="/code", env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    runs.commit()
    print(f"exit {r.returncode}: {job}\n" + log.read_text()[-1500:], flush=True)
    return r.returncode


@app.local_entrypoint()
def main(jobs: str):
    """`jobs` is a ';'-separated list of yambda_bench.py argument strings."""
    jobs = [j.strip() for j in jobs.split(";") if j.strip()]
    print(f"{len(jobs)} jobs on {GPU}")
    for j, c in zip(jobs, run_bench.map(jobs, return_exceptions=True)):
        print(c, j)
