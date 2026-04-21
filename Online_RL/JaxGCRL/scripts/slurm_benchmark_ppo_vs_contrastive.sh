#!/bin/bash
#SBATCH --job-name=ppo_benchmark
#SBATCH --account=mingyan
#SBATCH --partition=mingyan-a100
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/logs/ppo_benchmark-%A_%a.out
#SBATCH --error=/home/%u/logs/ppo_benchmark-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${PROFILE:-pilot}"

mkdir -p "/home/${USER}/logs"

eval "$(conda shell.bash hook)"
conda activate "${JAXGCRL_CONDA_ENV:-jaxgcrl}"

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

cd "${REPO_ROOT}"

benchmark_args=(
  --profile "${PROFILE}"
  --matrix-index "${SLURM_ARRAY_TASK_ID}"
)

if [[ "${LOG_WANDB:-0}" == "1" ]]; then
  benchmark_args+=(--log-wandb)
fi

python3 scripts/ppo_benchmark.py "${benchmark_args[@]}"
