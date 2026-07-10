#!/bin/bash
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=bed_main
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

set -euo pipefail

CONFIG_PATH="$1"
shift

export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME="$HOME/.cache/huggingface"
export XDG_CACHE_HOME="/scratch-ssd/$USER/.cache"
export TMPDIR="/scratch/$USER/tmp"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" slurm_logs

source /scratch-ssd/oatml/miniconda3/bin/activate 20_questions_env
if [ -f .env ]; then source .env; fi

echo "START TIME: $(date)"
srun python main.py -c "$CONFIG_PATH" "$@"
echo "END TIME: $(date)"
