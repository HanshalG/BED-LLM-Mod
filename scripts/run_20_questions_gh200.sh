#!/bin/bash
#SBATCH --partition=gh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=20_questions_EIG_animals
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

set -euo pipefail

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
    "$SINGULARITY_CACHEDIR" \
    "$SINGULARITY_TMPDIR" \
    "$HF_HOME" \
    "$TRANSFORMERS_CACHE" \
    "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$XDG_CACHE_HOME" \
    "$BED_LLM_PYDEPS"

echo "START TIME: $(date)"

singularity exec --nv \
    --bind "$PWD:$PWD,/scratch-ssd/$USER:/scratch-ssd/$USER" \
    --pwd "$PWD" \
    "$CONTAINER" bash -s "$1" << 'EOF'

set -euo pipefail

if [ -f .env ]; then
    source .env
fi

if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

python3 -m pip install --target "$BED_LLM_PYDEPS" --upgrade pyyaml wandb openai-harmony

python3 main.py -c configs/config$1.yaml

EOF

echo "END TIME: $(date)"
