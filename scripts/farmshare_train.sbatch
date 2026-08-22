#!/usr/bin/env bash
# One 124M training run on FarmShare's 4x L40S (this repo's original design
# target). No --account/--qos needed on FarmShare.
# VERIFY before first use (sinfo -o "%P %l %G"): the partition name and the
# gres label below are the common defaults and may differ.
# Usage (from the repo root):
#   sbatch scripts/farmshare_train.sbatch <config.yaml> [overrides...]
#SBATCH --job-name=attnres-train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG="${1:?usage: sbatch scripts/farmshare_train.sbatch <config.yaml> [overrides...]}"
shift
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"   # AFS home is tiny; keep caches beside the repo
NPROC="${SLURM_GPUS_ON_NODE:-4}"
.venv/bin/torchrun --standalone --nproc_per_node="${NPROC}" train.py --config "${CONFIG}" "$@"
