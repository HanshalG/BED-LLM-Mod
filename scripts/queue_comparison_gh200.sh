#!/bin/bash
# Queue all comparison configs on GH200 only (no MSC overflow).
set -euo pipefail

REPO="${REPO:-$HOME/BED-LLM-Mod-strategyeig-20trial-20260517T172146}"
GH200_SCRIPT="${REPO}/scripts/run_20_questions_gh200_qwen_triton.sh"
MANIFEST="${REPO}/slurm_logs/comparison_jobs.tsv"

cd "$REPO"
mkdir -p slurm_logs
if [[ ! -f "${MANIFEST}" ]]; then
  printf 'job_id\tconfig_id\tjob_name\tsubmitted_at\n' > "${MANIFEST}"
fi

submit_gh200() {
  local id="$1"
  local name="$2"
  echo "Queueing config${id} (${name}) on gh200..."
  local sbatch_out
  sbatch_out="$(sbatch -p gh200 --job-name="${name}" "${GH200_SCRIPT}" "${id}")"
  local job_id="${sbatch_out##* }"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${id}" "${name}" "$(date -Is)" >> "${MANIFEST}"
  echo "${sbatch_out}"
}

# Cancel MSC comparison relaunches
echo "Cancelling MSC jobs..."
scancel -u "$USER" -p msc 2>/dev/null || true

# Queue configs not already running on GH200 (skip if job name already RUNNING/PENDING on gh200)
running_configs() {
  squeue -u "$USER" -h -p gh200 -o '%j' 2>/dev/null | sed 's/qwen_//' || true
}

for entry in \
  "177:qwen_strat_d3" \
  "179:qwen_strat_d2_llm" \
  "180:qwen_strat_d3_llm"
do
  id="${entry%%:*}"
  name="${entry##*:}"
  if squeue -u "$USER" -h -p gh200 -o '%j' 2>/dev/null | grep -qx "${name}"; then
    echo "Skip config${id}: ${name} already in gh200 queue"
    continue
  fi
  submit_gh200 "${id}" "${name}"
done

echo ""
squeue -u "$USER" -o '%.10i %.9P %.24j %.8T %.10M %R'
