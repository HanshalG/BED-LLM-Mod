#!/bin/bash
#SBATCH --partition=llm
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bed_smoke_gemma4_4b
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#
# Cluster access:
#   ssh -J hanyal@cslinuxproxy hanyal@oat0.cs.ox.ac.uk
# Partitions on oat0: llm (LLM GPUs), gh200, msc, gb10 — there is no "a100" partition.
# Submit:
#   sbatch scripts/run_bed_smoke_a100.sh location

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_bed_smoke_gh200.sh" "${1:-location}"
