#!/bin/bash
#SBATCH --partition=msc,llm
#SBATCH --exclude=oat12
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=range_h3_stable
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err

set -euo pipefail

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
source /scratch-ssd/oatml/miniconda3/bin/activate 20_questions_env

echo "START TIME: $(date)"
srun python scripts/nonmyopic_range_gated_rock_stable_controls.py "$@"
echo "END TIME: $(date)"
