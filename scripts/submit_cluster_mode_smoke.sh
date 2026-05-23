#!/bin/bash
# Submit 8 small mode-smoke jobs: 3× gh200, 3× msc (A100), 2× llm (A100).
#
#   ssh -J hanyal@cslinuxproxy hanyal@oat0.cs.ox.ac.uk
#   cd ~/BED-LLM-Mod-refactor && bash scripts/submit_cluster_mode_smoke.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p slurm_logs

submit() {
  local partition="$1"
  local extra_sbatch="${2:-}"
  local config="$3"
  local job
  # shellcheck disable=SC2086
  job=$(sbatch -p "$partition" $extra_sbatch scripts/run_bed_smoke_cluster.sh "$config")
  echo "$job  partition=$partition  config=$config"
}

echo "Submitting 8 cluster mode-smoke jobs from $ROOT"

# GH200 (3): all 20 Questions modes
submit gh200 "--gpus=1 --cpus-per-task=72" "configs/cluster_smoke/01_animals_eig.yaml"
submit gh200 "--gpus=1 --cpus-per-task=72" "configs/cluster_smoke/02_animals_naive.yaml"
submit gh200 "--gpus=1 --cpus-per-task=72" "configs/cluster_smoke/03_animals_entropy_split.yaml"

# MSC A100 (3): all location-finding modes + animals EIG cross-check
submit msc "--gres=gpu:a100:1 --cpus-per-task=16" "configs/cluster_smoke/04_location_core.yaml"
submit msc "--gres=gpu:a100:1 --cpus-per-task=16" "configs/cluster_smoke/05_location_strategy.yaml"
submit msc "--gres=gpu:a100:1 --cpus-per-task=16" "configs/cluster_smoke/06_animals_eig_a100.yaml"

# LLM A100 (2): remaining 20 Questions modes
submit llm "--gres=gpu:a100:1 --cpus-per-task=16" "configs/cluster_smoke/07_animals_naive_belief_special.yaml"
submit llm "--gres=gpu:a100:1 --cpus-per-task=16" "configs/cluster_smoke/08_animals_strategy.yaml"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    slurm_logs/slurm-<jobid>.out"
