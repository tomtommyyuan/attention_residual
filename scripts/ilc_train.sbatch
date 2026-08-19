#!/usr/bin/env bash
# One training run on 4x A100-80G on the SNAP/ILC cluster.
# Usage (from the repo root on /dfs/user/$USER):
#   sbatch scripts/ilc_train.sbatch configs/baseline_124m.yaml [overrides...]
#SBATCH --job-name=attnres-train
#SBATCH --account=infolab
#SBATCH --partition=il
#SBATCH --qos=il
#SBATCH -N 1
#SBATCH --gres=gpu:a100:4
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG="${1:?usage: sbatch scripts/ilc_train.sbatch <config.yaml> [overrides...]}"
shift

# venv lives on the per-node SSD; rebuild if this node doesn't have it yet
# (pip cache is also per-node, so a rebuild costs a few minutes once per node)
export XDG_CACHE_HOME="/lfs/local/0/${USER}/.cache"
export PIP_CACHE_DIR="/lfs/local/0/${USER}/.cache/pip"
export HF_HOME="/lfs/local/0/${USER}/.cache/huggingface"
# pick the torch build to match the allocated GPU: B200 (sm_100) needs the
# CUDA-13 torch from PyPI (drivers 580+); older cards run cu124 wheels
if nvidia-smi -L 2>/dev/null | grep -q "B200"; then
  VENV="/lfs/local/0/${USER}/venvs/attnres-cu13"
  TORCH_SPEC=(torch)
else
  VENV="/lfs/local/0/${USER}/venvs/attnres"
  TORCH_SPEC=(torch --index-url https://download.pytorch.org/whl/cu124)
fi
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
if [ ! -d "${VENV}" ]; then
  "${PY}" -m venv "${VENV}"
  "${VENV}/bin/pip" install --upgrade pip
fi
"${VENV}/bin/pip" install --only-binary :all: "${TORCH_SPEC[@]}"
"${VENV}/bin/pip" install --only-binary :all: -r requirements.txt

# world size follows the actual allocation, so `--gres=gpu:<type>:2` jobs work
# (keep the GLOBAL batch fixed across compared runs: world * grad_accum = const)
NPROC="${SLURM_GPUS_ON_NODE:-4}"
"${VENV}/bin/torchrun" --standalone --nproc_per_node="${NPROC}" train.py --config "${CONFIG}" "$@"
