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

CONTAINER="docker://vllm/vllm-openai:v0.23.0"

export SINGULARITY_CACHEDIR=/scratch-ssd/$USER/cache
export SINGULARITY_TMPDIR=/scratch-ssd/$USER/tmp
export APPTAINER_TMPDIR=/scratch-ssd/$USER/tmp
export TMPDIR=/scratch-ssd/$USER/tmp
export HF_HOME=/scratch-ssd/$USER/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export BED_LLM_PYDEPS=/scratch-ssd/$USER/bed-llm-pydeps-vllm-0.23.0
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BED_LLM_PYDEPS"
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
if [[ -z "${BED_LLM_VLLM_KWARGS:-}" ]]; then
  export BED_LLM_VLLM_KWARGS='{"max_num_seqs":4,"enforce_eager":false}'
fi

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

srun singularity exec --nv \
  --bind "$PWD:$PWD,/scratch-ssd/$USER:/scratch-ssd/$USER" \
  --pwd "$PWD" \
  "$CONTAINER" bash -s -- "$@" <<'EOF'
set -euo pipefail

if [[ -f .env ]]; then
  source .env
fi
if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

if ! python3 -c 'import openai_harmony, pomdp_py, yaml' >/dev/null 2>&1; then
  python3 -m pip install --target "$BED_LLM_PYDEPS" --upgrade --no-deps \
    openai-harmony pomdp-py==1.3.5.1 pyyaml
fi
python3 -c 'import pomdp_py; from environments.rock_diagnosis import get_paper_map; assert len(get_paper_map("15-15").rock_positions) == 15'

exec python3 "$@"
EOF

echo "END TIME: $(date)"
