#!/usr/bin/env bash
# One-time setup on the SNAP/ILC cluster: venv + deps + tests + FineWeb-Edu.
# Submit from the repo root (which should live on /dfs/user/$USER -- shared
# across nodes; /sailhome is only 20G): sbatch scripts/ilc_prepare.sbatch
#SBATCH --job-name=attnres-prep
#SBATCH --account=infolab
#SBATCH --partition=il
#SBATCH --qos=il
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o slurm-%j-prep.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# /lfs/local/0 is fast per-node SSD: venvs and caches go here, NEVER /sailhome.
# The venv is per-node, so every job script re-checks and builds if missing.
export XDG_CACHE_HOME="/lfs/local/0/${USER}/.cache"
export PIP_CACHE_DIR="/lfs/local/0/${USER}/.cache/pip"
export HF_HOME="/lfs/local/0/${USER}/.cache/huggingface"
VENV="/lfs/local/0/${USER}/venvs/attnres"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
if [ ! -d "${VENV}" ]; then
  "${PY}" -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
fi
# cu124 torch (drivers on Stanford clusters are 550-series; PyPI default torch
# now ships CUDA-13 builds) + wheels only (no build toolchains on HPC nodes)
"${VENV}/bin/pip" install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cu124
"${VENV}/bin/pip" install --only-binary :all: -r requirements.txt
"${VENV}/bin/pytest" tests/ -q
"${VENV}/bin/python" data/prepare_fineweb_edu.py --nproc "${SLURM_CPUS_PER_TASK}"
