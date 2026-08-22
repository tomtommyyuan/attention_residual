#!/usr/bin/env bash
# Bootstrap a fresh Azure GPU VM for training.
# Assumes an image with the NVIDIA driver preinstalled (recommended:
# microsoft-dsvm:ubuntu-hpc:2204:latest). Run as the login user.
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git tmux

cd ~
[ -d LLM_training ] || git clone https://github.com/tomtommyyuan/attention_residual.git LLM_training
cd LLM_training

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
# cu124 torch: matches every checkpoint we produced on the Stanford clusters
.venv/bin/pip install -q --only-binary :all: torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -q --only-binary :all: -r requirements.txt
.venv/bin/pytest tests/ -q

cat <<'EONOTE'
bootstrap done. Next steps:
  data (pick the set the run needs):
    124M set:  .venv/bin/python data/prepare_fineweb_edu.py
    350M/760M: .venv/bin/python data/prepare_fineweb_edu.py \
                 --out_dir data/fineweb_edu_100bt --remote_name sample-100BT \
                 --train_tokens 15500000000
  resume from HF backup (optional):
    .venv/bin/pip install -q huggingface_hub
    .venv/bin/python -c "from huggingface_hub import hf_hub_download; \
      hf_hub_download('tomyuanyucheng/attention-residual-ckpts', '<run>/latest.pt', local_dir='hf_ckpts')"
  train (spot-resilient):
    tmux new -d -s train './scripts/azure_train_loop.sh configs/<cfg>.yaml 1 train.grad_accum_steps=16'
EONOTE
