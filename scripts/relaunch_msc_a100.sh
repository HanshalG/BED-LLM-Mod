#!/bin/bash
# Relaunch StrategyEIG jobs on MSC A100 (idle nodes, lower GPU util configs).
set -euo pipefail

REPO="${REPO:-$HOME/BED-LLM-Mod-strategyeig-20trial-20260517T172146}"
A100_SCRIPT="${REPO}/scripts/run_20_questions_qwen_a100.sh"
MANIFEST="${REPO}/slurm_logs/comparison_jobs.tsv"

cd "$REPO"
mkdir -p slurm_logs

submit_msc() {
  local config_id="$1"
  local job_name="$2"
  local nodelist="$3"
  echo "Submitting config${config_id} (${job_name}) on msc/a100 nodelist=${nodelist}"
  local sbatch_out
  sbatch_out="$(
    sbatch -p msc --gres=gpu:a100:1 --cpus-per-task=16 \
      -w "${nodelist}" --job-name="${job_name}" \
      "${A100_SCRIPT}" "${config_id}"
  )"
  local job_id="${sbatch_out##* }"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${config_id}" "${job_name}" "$(date -Is)" >> "${MANIFEST}"
  echo "${sbatch_out}"
}

# Cancel pending GH200 strategy d3 if still queued
if squeue -h -j 87417 2>/dev/null | grep -q .; then
  echo "Cancelling pending GH200 job 87417"
  scancel 87417
fi

# config181 = strategy d3 analytical (was 177 on gh200)
# config182 = strategy d2 llm (retry 179)
# config183 = strategy d3 llm (retry 180)
submit_msc 181 qwen_strat_d3_msc oat10
submit_msc 182 qwen_strat_d2_llm_msc oat17
submit_msc 183 qwen_strat_d3_llm_msc oat17

echo ""
squeue -u "$USER" -o '%.10i %.9P %.24j %.8T %.10M %R' | head -15
