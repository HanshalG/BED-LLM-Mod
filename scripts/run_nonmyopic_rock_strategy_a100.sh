#!/bin/bash
#SBATCH --partition=msc,llm
#SBATCH --exclude=oat12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

set -euo pipefail

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/transformers
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets
export HF_HOME=$HOME/.cache/huggingface
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export TMPDIR=/scratch/$USER/tmp
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
if [[ -z "${BED_LLM_VLLM_KWARGS:-}" ]]; then
  export BED_LLM_VLLM_KWARGS='{"max_num_seqs":4,"enforce_eager":false}'
fi

mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" slurm_logs
source /scratch-ssd/oatml/miniconda3/bin/activate 20_questions_env
source .env

if ! python -c 'import pomdp_py' >/dev/null 2>&1; then
  /scratch-ssd/oatml/run_locked.sh \
    python -m pip install --no-cache-dir pomdp-py==1.3.5.1
fi
python -c 'import pomdp_py; from environments.rock_diagnosis import get_paper_map; assert len(get_paper_map("15-15").rock_positions) == 15'

exec srun python "$@"
