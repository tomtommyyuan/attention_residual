#!/usr/bin/env bash
# One training run on 4 GPUs of the HAI cluster.
# Usage (from the repo root):
#   sbatch scripts/slurm_train.sbatch configs/baseline_124m.yaml [overrides...]
# GPU type is pinned to h100 so all compared runs share identical hardware.
#SBATCH --job-name=attnres-train
#SBATCH --account=ingrai
#SBATCH -p hai
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG="${1:?usage: sbatch scripts/slurm_train.sbatch <config.yaml> [overrides...]}"
shift
export HF_HOME="/hai/scratch/${USER}/hf_cache"
source .venv/bin/activate
torchrun --standalone --nproc_per_node=4 train.py --config "${CONFIG}" "$@"
