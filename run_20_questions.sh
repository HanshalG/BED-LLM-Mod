#!/bin/bash
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=20_questions_EIG_animals
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Tell conda to use fast local storage
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME=$HOME/.cache/huggingface

# Redirect all user caches away from ~/.cache - I get en error otherwise
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
mkdir -p "$XDG_CACHE_HOME"

# Ensure TMPDIR is writable on compute node
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR"

# Ensure conda directories exist
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

rm -rf ~/.cache/pip
export PIP_NO_CACHE_DIR=1

# exports to make vLLM work
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Nuke the existing environment
# /scratch-ssd/oatml/run_locked.sh \
#   /scratch-ssd/oatml/miniconda3/bin/conda env remove -n 20_questions_env -y
# rm -rf "$CONDA_ENVS_PATH/20_questions_env"

# Create or update the environment from environment.yml
/scratch-ssd/oatml/run_locked.sh \
  /scratch-ssd/oatml/miniconda3/bin/conda env update -f environment.yml

# Activate the environment
source /scratch-ssd/oatml/miniconda3/bin/activate 20_questions_env


pip install accelerate

if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
else
    echo "Warning: HUGGINGFACE_TOKEN is not set. Skipping Hugging Face login."
fi
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "Warning: WANDB_API_KEY is not set. Skipping wandb login."
fi

echo "START TIME: $(date)"

# Run the script
srun python main.py -c config.yaml

echo "END TIME: $(date)"