#!/bin/bash
#SBATCH --partition=gh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=20_questions_EIG_animals
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

export GH200_CONTAINER=docker://nvcr.io/nvidia/pytorch:24.07-py3

# Tell singularity + pip to use fast local storage
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME=$HOME/.cache/huggingface
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export TMPDIR=/scratch-ssd/$USER/tmp
export SINGULARITY_CACHEDIR=/scratch-ssd/$USER/singularity-cache
export SINGULARITY_TMPDIR=/scratch-ssd/$USER/tmp
export APPTAINER_TMPDIR=/scratch-ssd/$USER/tmp
export PYTHONUSERBASE=/scratch-ssd/$USER/python-gh200
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$SINGULARITY_CACHEDIR" "$PYTHONUSERBASE"

rm -rf ~/.cache/pip
export PIP_NO_CACHE_DIR=1

# exports to make vLLM work
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source .env

if [ -n "$HUGGINGFACE_TOKEN" ]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export SINGULARITYENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_PYTHONUSERBASE="$PYTHONUSERBASE"
export SINGULARITYENV_PATH="$PYTHONUSERBASE/bin:$PATH"
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING="$TORCH_NCCL_ASYNC_ERROR_HANDLING"
export SINGULARITYENV_NCCL_DEBUG="$NCCL_DEBUG"
export SINGULARITYENV_NCCL_ASYNC_ERROR_HANDLING="$NCCL_ASYNC_ERROR_HANDLING"
export SINGULARITYENV_NCCL_BLOCKING_WAIT="$NCCL_BLOCKING_WAIT"
export SINGULARITYENV_NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE"
export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD"
export SINGULARITYENV_HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"
export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"

BIND_PATHS="$PWD:$PWD,$TMPDIR:$TMPDIR,$XDG_CACHE_HOME:$XDG_CACHE_HOME,$TRANSFORMERS_CACHE:$TRANSFORMERS_CACHE,$HF_HUB_CACHE:$HF_HUB_CACHE,$HF_DATASETS_CACHE:$HF_DATASETS_CACHE,$PYTHONUSERBASE:$PYTHONUSERBASE"

singularity exec --nv \
  --bind "$BIND_PATHS" \
  --pwd "$PWD" \
  "$GH200_CONTAINER" \
  python -m pip install --user --upgrade-strategy only-if-needed pyyaml wandb accelerate openai-harmony transformers vllm

echo "START TIME: $(date)"

# Run the script
srun singularity exec --nv \
  --bind "$BIND_PATHS" \
  --pwd "$PWD" \
  "$GH200_CONTAINER" \
  python main.py -c config.yaml

echo "END TIME: $(date)"
