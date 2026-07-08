#!/bin/bash
#SBATCH --partition=gh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=strategy_ranking_fidelity
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

set -euo pipefail

CONFIG_ARG="${1:?Path or numeric config id required}"
shift

if [ -f "$CONFIG_ARG" ]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="configs/config${CONFIG_ARG}.yaml"
fi

CONTAINER="docker://vllm/vllm-openai:gemma4"

export SINGULARITY_CACHEDIR=/scratch-ssd/$USER/cache
export SINGULARITY_TMPDIR=/scratch-ssd/$USER/tmp
export APPTAINER_TMPDIR=/scratch-ssd/$USER/tmp
export TMPDIR=/scratch-ssd/$USER/tmp
export VLLM_ENABLE_CUDA_COMPATIBILITY=1

export HF_HOME=/scratch-ssd/$USER/huggingface
export TRANSFORMERS_CACHE=/scratch-ssd/$USER/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/$USER/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/$USER/huggingface/datasets
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export BED_LLM_PYDEPS=/scratch-ssd/$USER/bed-llm-pydeps
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BED_LLM_PYDEPS"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p \
    slurm_logs \
    "$SINGULARITY_CACHEDIR" \
    "$SINGULARITY_TMPDIR" \
    "$HF_HOME" \
    "$TRANSFORMERS_CACHE" \
    "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$XDG_CACHE_HOME" \
    "$BED_LLM_PYDEPS"

echo "START TIME: $(date)"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "ARGS=$*"

singularity exec --nv \
    --bind "$PWD:$PWD,/scratch-ssd/$USER:/scratch-ssd/$USER" \
    --pwd "$PWD" \
    "$CONTAINER" bash -s -- "$CONFIG_PATH" "$@" << 'EOF'

set -euo pipefail

CONFIG_PATH="$1"
shift

if [ -f .env ]; then
    source .env
fi

if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

python3 -m pip install --target "$BED_LLM_PYDEPS" --upgrade pyyaml wandb openai-harmony

python3 scripts/strategy_ranking_fidelity.py -c "$CONFIG_PATH" "$@"

EOF

echo "END TIME: $(date)"
