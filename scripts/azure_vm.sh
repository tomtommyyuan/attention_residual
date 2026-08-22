#!/usr/bin/env bash
# Azure VM helpers for training boxes (single-GPU A100 spot by default).
# Usage:
#   ./scripts/azure_vm.sh create <name> [size] [spot|ondemand]
#   ./scripts/azure_vm.sh ssh <name>
#   ./scripts/azure_vm.sh stop <name>        # deallocate: GPU billing stops
#   ./scripts/azure_vm.sh start <name>
#   ./scripts/azure_vm.sh delete <name>
set -euo pipefail
RG="attnres"
LOC="westus2"
CMD="${1:?create|ssh|stop|start|delete}"
NAME="${2:?vm name}"
SIZE="${3:-Standard_NC24ads_A100_v4}"
MODE="${4:-spot}"

case "$CMD" in
  create)
    az group create -n "$RG" -l "$LOC" -o none
    SPOT_ARGS=()
    if [ "$MODE" = "spot" ]; then
      SPOT_ARGS=(--priority Spot --eviction-policy Deallocate --max-price -1)
    fi
    az vm create -g "$RG" -n "$NAME" \
      --image microsoft-dsvm:ubuntu-hpc:2204:latest \
      --size "$SIZE" "${SPOT_ARGS[@]}" \
      --admin-username tom --generate-ssh-keys \
      --os-disk-size-gb 256 --public-ip-sku Standard -o table
    echo "next: ./scripts/azure_vm.sh ssh $NAME  ->  bash <(curl -sL https://raw.githubusercontent.com/tomtommyyuan/attention_residual/main/scripts/azure_bootstrap.sh)"
    ;;
  ssh)    az ssh vm -g "$RG" -n "$NAME" ;;
  stop)   az vm deallocate -g "$RG" -n "$NAME" -o table ;;
  start)  az vm start -g "$RG" -n "$NAME" -o table ;;
  delete) az vm delete -g "$RG" -n "$NAME" --yes -o table ;;
  *) echo "unknown command $CMD" >&2; exit 1 ;;
esac
