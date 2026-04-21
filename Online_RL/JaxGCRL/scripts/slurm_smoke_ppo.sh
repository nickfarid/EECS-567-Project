#!/bin/bash
#SBATCH --job-name=ppo_smoke
#SBATCH --account=mingyan
#SBATCH --partition=mingyan-a100
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/logs/ppo_smoke-%j.out
#SBATCH --error=/home/%u/logs/ppo_smoke-%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

mkdir -p "/home/${USER}/logs"

eval "$(conda shell.bash hook)"
conda activate "${JAXGCRL_CONDA_ENV:-jaxgcrl}"
set -u

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

cd "${REPO_ROOT}"

for agent in ppo ppo_contrastive; do
  for env in reacher ant; do
    python3 "${REPO_ROOT}/scripts/ppo_benchmark.py" --profile smoke --agent "${agent}" --env "${env}" --seed 0
  done
done
