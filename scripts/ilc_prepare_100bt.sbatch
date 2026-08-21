#!/usr/bin/env bash
# Phase-C dataset: 15.5B FineWeb-Edu tokens from sample-100BT into a separate
# dir (serves both the 350M/7.2B and 760M/15.2B token budgets; the 124M
# dataset in data/fineweb_edu stays untouched). CPU-only -> il-cpu partition.
# Submit from the repo root: sbatch scripts/ilc_prepare_100bt.sbatch
#SBATCH --job-name=attnres-prep100
#SBATCH --account=infolab
#SBATCH --partition=il-cpu
#SBATCH --qos=il
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH -o slurm-%j-prep100.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/lfs/local/0/${USER}/.cache"
export PIP_CACHE_DIR="/lfs/local/0/${USER}/.cache/pip"
export HF_HOME="/lfs/local/0/${USER}/.cache/huggingface"

# il-cpu nodes have their own /lfs -> build the venv if this node lacks one
VENV="/lfs/local/0/${USER}/venvs/attnres"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
if [ ! -d "${VENV}" ]; then
  "${PY}" -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
fi
"${VENV}/bin/pip" install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cu124
"${VENV}/bin/pip" install --only-binary :all: -r requirements.txt

"${VENV}/bin/python" data/prepare_fineweb_edu.py \
  --out_dir data/fineweb_edu_100bt \
  --remote_name sample-100BT \
  --train_tokens 15500000000 \
  --nproc "${SLURM_CPUS_PER_TASK}"
