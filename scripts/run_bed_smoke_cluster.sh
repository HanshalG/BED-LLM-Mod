#!/bin/bash
#SBATCH --job-name=bed_mode_smoke
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#
# Usage (partition is chosen at submit time):
#   sbatch -p gh200 --cpus-per-task=72 scripts/run_bed_smoke_cluster.sh configs/cluster_smoke/01_animals_eig.yaml
#   sbatch -p msc --gres=gpu:a100:1 --cpus-per-task=16 scripts/run_bed_smoke_cluster.sh configs/cluster_smoke/04_location_core.yaml
#   sbatch -p llm --cpus-per-task=16 scripts/run_bed_smoke_cluster.sh configs/cluster_smoke/07_animals_special.yaml

set -euo pipefail

CONFIG="${1:?Path to YAML config (e.g. configs/cluster_smoke/01_animals_eig.yaml)}"
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

echo "START TIME: $(date) partition=${SLURM_JOB_PARTITION:-unknown} config=$CONFIG job=$SLURM_JOB_ID"

singularity exec --nv \
    --bind "$PWD:$PWD,/scratch-ssd/$USER:/scratch-ssd/$USER" \
    --pwd "$PWD" \
    "$CONTAINER" bash -s << EOF
set -euo pipefail
if [ -f .env ]; then source .env; fi
if [ -n "\${HUGGINGFACE_TOKEN:-}" ]; then export HF_TOKEN="\$HUGGINGFACE_TOKEN"; fi
rm -rf "\$BED_LLM_PYDEPS"
mkdir -p "\$BED_LLM_PYDEPS"
python3 -m pip install --target "\$BED_LLM_PYDEPS" --no-cache-dir pyyaml wandb openai-harmony
python3 main.py -c "$CONFIG" --output-root runs/cluster_smoke --run-name "\$(basename "$CONFIG" .yaml)"
EOF

echo "END TIME: $(date)"
