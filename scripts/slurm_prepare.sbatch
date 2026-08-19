#!/usr/bin/env bash
# One-time setup on the HAI cluster: venv + deps + tests + FineWeb-Edu shards.
# CPU-only, so it goes to the roomy preemptible partition.
# Submit from the repo root: sbatch scripts/slurm_prepare.sbatch
#SBATCH --job-name=attnres-prep
#SBATCH --account=ingrai
#SBATCH -p hai-lo
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o slurm-%j-prep.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="/hai/scratch/${USER}/hf_cache"   # keep caches off the 50G home quota

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
fi
# torch pinned to the cu124 index: cluster drivers are 550-series (CUDA 12.4),
# and the default PyPI torch now ships CUDA-13 builds that need driver >= 580
.venv/bin/pip install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install --only-binary :all: -r requirements.txt
.venv/bin/pytest tests/ -q
.venv/bin/python data/prepare_fineweb_edu.py --nproc "${SLURM_CPUS_PER_TASK}"
