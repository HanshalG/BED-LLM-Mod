#!/bin/bash
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --job-name=range_gated_h3
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

set -euo pipefail

CONFIG_PATH="${1:?usage: run_range_gated_fixed_tail_cluster.sh CONFIG [extra args]}"
shift
ENTRYPOINT="${BED_LLM_RANGE_GATED_ENTRYPOINT:-scripts/nonmyopic_range_gated_rock_fixed_tail.py}"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME=$HOME/.cache/huggingface
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export TMPDIR=/scratch/$USER/tmp
export PIP_NO_CACHE_DIR=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" slurm_logs

if [ "${BED_LLM_SKIP_ENV_SETUP:-0}" = "1" ]; then
  echo "BED_LLM_SKIP_ENV_SETUP=1; skipping conda/pip environment setup."
else
  /scratch-ssd/oatml/run_locked.sh \
    /scratch-ssd/oatml/miniconda3/bin/conda env update -f environment.yml
  /scratch-ssd/oatml/run_locked.sh \
    pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129
  /scratch-ssd/oatml/run_locked.sh \
    pip install --upgrade \
    "https://wheels.vllm.ai/c0c98b8b9a392c7e8b36b68cf477e245dda48d80/vllm-0.19.1rc1.dev367%2Bgc0c98b8b9-cp38-abi3-manylinux_2_31_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
  /scratch-ssd/oatml/run_locked.sh \
    pip install --ignore-installed --no-deps transformers==5.5.0
fi

source /scratch-ssd/oatml/miniconda3/bin/activate 20_questions_env
if [ -f .env ]; then
  source .env
fi
if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
  huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

echo "START TIME: $(date)"
srun python "$ENTRYPOINT" \
  --config "$CONFIG_PATH" \
  "$@"
echo "END TIME: $(date)"
