#!/usr/bin/env bash
# One-time setup on Sherlock: venv + deps + tests + FineWeb-Edu shards (CPU only).
# Submit from the repo root: sbatch scripts/sherlock_prepare.sbatch
#SBATCH --job-name=attnres-prep
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o slurm-%j-prep.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
ml python/3.12.1
export HF_HOME="${GROUP_HOME}/${USER}/hf_cache"   # keep caches off the 15G home quota

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
fi
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest tests/ -q
.venv/bin/python data/prepare_fineweb_edu.py --nproc "${SLURM_CPUS_PER_TASK}"
