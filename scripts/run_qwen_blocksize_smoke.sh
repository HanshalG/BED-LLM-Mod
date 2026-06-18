#!/bin/bash
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=qwen_block_smoke
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

set -euo pipefail

BLOCK_SIZE="${1:?usage: run_qwen_blocksize_smoke.sh BLOCK_SIZE [BATCH_SIZE] [MAX_NEW_TOKENS]}"
BATCH_SIZE="${2:-$BLOCK_SIZE}"
MAX_NEW_TOKENS="${3:-32}"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME=$HOME/.cache/huggingface
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" slurm_logs

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}'
export WANDB_MODE=disabled

source /scratch-ssd/oatml/miniconda3/bin/activate 20_questions_env
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "START TIME: $(date)"
echo "BLOCK_SIZE=$BLOCK_SIZE BATCH_SIZE=$BATCH_SIZE MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
srun python scripts/test_qwen_batched_block_size.py \
  --block-size "$BLOCK_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS"
echo "END TIME: $(date)"
