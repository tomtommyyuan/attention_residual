#!/usr/bin/env bash
# Launch the Full AttnRes run on 4 GPUs. Extra key=value args are forwarded as
# config overrides, e.g.: ./scripts/launch_attnres.sh train.seed=42
set -euo pipefail
cd "$(dirname "$0")/.."
torchrun --standalone --nproc_per_node=4 train.py --config configs/attnres_124m.yaml "$@"
