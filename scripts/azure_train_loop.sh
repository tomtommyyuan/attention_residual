#!/usr/bin/env bash
# Spot-resilient training runner: (re)starts train.py, resuming from
# latest.pt whenever it exists, until the run completes. Pair with a
# @reboot cron/tmux line so a deallocated spot VM resumes on restart:
#   (crontab -l; echo "@reboot cd ~/LLM_training && tmux new -d -s train \
#     './scripts/azure_train_loop.sh configs/foo.yaml 1 ...'") | crontab -
# Usage: ./scripts/azure_train_loop.sh <config.yaml> <nproc> [overrides...]
set -uo pipefail
CONFIG="${1:?usage: azure_train_loop.sh <config.yaml> <nproc> [overrides...]}"
NPROC="${2:?nproc required}"
shift 2
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

RUN_NAME=$(.venv/bin/python - "$CONFIG" "$@" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
name = cfg["train"]["run_name"]
for a in sys.argv[2:]:
    if a.startswith("train.run_name="):
        name = a.split("=", 1)[1]
print(name)
EOF
)
OUT="out/${RUN_NAME}"
echo "run: ${RUN_NAME} (resume file: ${OUT}/latest.pt)"

for attempt in $(seq 1 100); do
  RESUME=()
  [ -f "${OUT}/latest.pt" ] && RESUME=(--resume "${OUT}/latest.pt")
  .venv/bin/torchrun --standalone --nproc_per_node="${NPROC}" train.py \
    --config "${CONFIG}" "$@" "${RESUME[@]}"
  code=$?
  if [ "${code}" -eq 0 ]; then
    echo "run complete after ${attempt} attempt(s)"
    exit 0
  fi
  echo "attempt ${attempt} exited with ${code}; retrying in 30s"
  sleep 30
done
echo "gave up after 100 attempts" >&2
exit 1
