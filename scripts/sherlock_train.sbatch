#!/usr/bin/env bash
# One training run on 4 data-center GPUs on Sherlock.
# Usage (from the repo root):
#   sbatch scripts/sherlock_train.sbatch configs/baseline_124m.yaml [overrides...]
#
# The GPU_MEM:80GB constraint targets A100-80G/H100-class cards and keeps the
# run off pre-Ampere GPUs (bf16 autocast needs compute capability >= 8.0).
# List what the gpu partition actually offers with `node_feat -p gpu` (or
# `sinfo -p gpu -O nodelist,features:80`) and adjust if the queue is long --
# all compared runs should use the SAME constraint for identical hardware.
#SBATCH --job-name=attnres-train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -G 4
#SBATCH -C GPU_MEM:80GB
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1   # stdout is block-buffered under slurm redirection
CONFIG="${1:?usage: sbatch scripts/sherlock_train.sbatch <config.yaml> [overrides...]}"
shift
ml python/3.12.1
export HF_HOME="${GROUP_HOME}/${USER}/hf_cache"
source .venv/bin/activate
torchrun --standalone --nproc_per_node=4 train.py --config "${CONFIG}" "$@"
