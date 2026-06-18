#!/bin/bash
# SSH to oat0 and run cluster batch monitor (uses ProxyJump from ~/.ssh/config).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_HOST="${SSH_HOST:-oat0}"
REMOTE_REPO="${REMOTE_REPO:-\$HOME/BED-LLM-Mod-strategyeig-20trial-20260517T172146}"

ssh -o BatchMode=yes "${SSH_HOST}" \
  "REPO=${REMOTE_REPO} bash ${REMOTE_REPO}/scripts/monitor_cluster_batch.sh"
