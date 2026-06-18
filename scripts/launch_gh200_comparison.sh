#!/bin/bash
# Submit EIG + StrategyEIG comparison jobs on GH200 (run from oat0 login).
set -euo pipefail

REPO="${REPO:-$HOME/BED-LLM-Mod-strategyeig-20trial-20260517T172146}"
SCRIPT="${REPO}/scripts/run_20_questions_gh200_qwen_triton.sh"

cd "$REPO"
mkdir -p slurm_logs

submit() {
  local id="$1"
  local name="$2"
  echo "Submitting config${id} (${name})..."
  sbatch --job-name="${name}" "$SCRIPT" "${id}"
}

# EIG depth 1 — analytical + LLM posterior (num_mc_samples=24)
submit 174 qwen_eig_d1_mc24
submit 178 qwen_eig_d1_llm_mc24

# StrategyEIG planning depth 2 & 3 — analytical + LLM posterior
submit 176 qwen_strat_d2
submit 177 qwen_strat_d3
submit 179 qwen_strat_d2_llm
submit 180 qwen_strat_d3_llm

echo ""
squeue -u "$USER" | head -12
