#!/bin/bash
# Submit short framework smoke runs for configs/config901.yaml ... config908.yaml.
#
# Run this on oat0 from the repo root:
#   bash scripts/submit_framework_smoke_901_908.sh

set -euo pipefail

MAX_ACTIVE="${MAX_ACTIVE:-8}"
CONFIG_IDS=(901 902 903 904 905 906 907 908)

mkdir -p slurm_logs

active_jobs() {
  squeue -h -u "$USER" -t PENDING,RUNNING,CONFIGURING,COMPLETING 2>/dev/null | wc -l | tr -d ' '
}

gh200_available() {
  sinfo -h -p gh200 -t idle,mix -o "%D" 2>/dev/null | awk '{total += $1} END {print total + 0}'
}

submit_one() {
  local config_id="$1"
  local active
  while true; do
    active="$(active_jobs)"
    if [ "$active" -lt "$MAX_ACTIVE" ]; then
      break
    fi
    echo "Already have $active active jobs; waiting for below MAX_ACTIVE=$MAX_ACTIVE"
    sleep 60
  done

  if [ "$(gh200_available)" -gt 0 ]; then
    echo "Submitting config${config_id}.yaml to gh200"
    sbatch scripts/run_20_questions_gh200.sh "$config_id"
  else
    echo "Submitting config${config_id}.yaml to msc"
    sbatch scripts/run_20_questions.sh "$config_id"
  fi
}

for config_id in "${CONFIG_IDS[@]}"; do
  submit_one "$config_id"
done

echo
echo "Submitted framework smoke matrix. Monitor with:"
echo "  squeue -u $USER -o '%.18i %.9P %.30j %.8T %.10M %.6D %R'"
echo "  tail -f slurm_logs/slurm-<jobid>.out"
