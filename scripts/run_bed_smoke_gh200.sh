#!/bin/bash
#SBATCH --partition=gh200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=bed_smoke_gemma4_4b
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#
# Cluster access:
#   ssh -J hanyal@cslinuxproxy hanyal@oat0.cs.ox.ac.uk
# Submit:
#   sbatch scripts/run_bed_smoke_gh200.sh location
#   sbatch scripts/run_bed_smoke_gh200.sh animals

set -euo pipefail

TASK="${1:-location}"
if [ "$TASK" = "location" ]; then
  CONFIG="config_smoke_location_gemma4_4b.yaml"
else
  CONFIG="config_smoke_animals_gemma4_4b.yaml"
fi

CONTAINER="docker://vllm/vllm-openai:gemma4"

export SINGULARITY_CACHEDIR=/scratch-ssd/$USER/cache
export SINGULARITY_TMPDIR=/scratch-ssd/$USER/tmp
export APPTAINER_TMPDIR=/scratch-ssd/$USER/tmp
export TMPDIR=/scratch-ssd/$USER/tmp
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export HF_HOME=/scratch-ssd/$USER/huggingface
export BED_LLM_PYDEPS=/scratch-ssd/$USER/bed-llm-pydeps
export PYTHONNOUSERSITE=1
export PYTHONPATH="$BED_LLM_PYDEPS"

mkdir -p slurm_logs "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$HF_HOME" "$BED_LLM_PYDEPS"

echo "START TIME: $(date) task=$TASK config=$CONFIG"

singularity exec --nv \
    --bind "$PWD:$PWD,/scratch-ssd/$USER:/scratch-ssd/$USER" \
    --pwd "$PWD" \
    "$CONTAINER" bash -s << EOF
set -euo pipefail
if [ -f .env ]; then source .env; fi
if [ -n "\${HUGGINGFACE_TOKEN:-}" ]; then export HF_TOKEN="\$HUGGINGFACE_TOKEN"; fi
python3 -m pip install --target "\$BED_LLM_PYDEPS" --upgrade pyyaml wandb openai-harmony
python3 main.py -c "$CONFIG" --output-root runs/smoke
EOF

echo "END TIME: $(date)"
