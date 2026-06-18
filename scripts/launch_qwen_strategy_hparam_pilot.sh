#!/bin/bash
# Launch an 8-job one-trial Qwen StrategyEIG hyperparameter pilot.
set -euo pipefail

REPO="${REPO:-$HOME/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z}"
RUN_SCRIPT="${RUN_SCRIPT:-$REPO/scripts/run_20_questions.sh}"
MANIFEST="${MANIFEST:-$REPO/slurm_logs/qwen_strategy_hparam_pilot.tsv}"

cd "$REPO"
mkdir -p slurm_logs

export BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}'

printf "job_id\tconfig_id\tname\tsubmitted_at\n" > "$MANIFEST"

submit() {
  local config_id="$1"
  local name="$2"
  local sbatch_out
  sbatch_out="$(
    sbatch -p msc --gres=gpu:a100:1 --cpus-per-task=16 \
      --job-name="$name" \
      "$RUN_SCRIPT" "$config_id"
  )"
  local job_id="${sbatch_out##* }"
  printf "%s\t%s\t%s\t%s\n" "$job_id" "$config_id" "$name" "$(date -Is)" >> "$MANIFEST"
  echo "$sbatch_out"
}

submit 1000 qwen_hp_d2_g10_b30_r4
submit 1001 qwen_hp_d3_g10_b30_r4
submit 1002 qwen_hp_d2_g20_b60_r4
submit 1003 qwen_hp_d3_g20_b60_r4
submit 1004 qwen_hp_d2_g10_b30_r8
submit 1005 qwen_hp_d3_g10_b30_r8
submit 1006 qwen_hp_d2_g20_b60_r8
submit 1007 qwen_hp_d3_g20_b60_r8

echo ""
squeue -u "$USER" -o "%.10i %.9P %.24j %.8T %.10M %R" | head -30
