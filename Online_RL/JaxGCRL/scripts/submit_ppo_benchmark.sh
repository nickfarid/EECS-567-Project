#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="${1:-}"
LOG_WANDB=0

if [[ -z "${PROFILE}" ]]; then
  echo "Usage: $0 {pilot|final} [--log-wandb]"
  exit 1
fi

if [[ "${PROFILE}" != "pilot" && "${PROFILE}" != "final" ]]; then
  echo "Profile must be one of: pilot, final"
  exit 1
fi

if [[ "${2:-}" == "--log-wandb" ]]; then
  LOG_WANDB=1
fi

mkdir -p "/home/${USER}/logs"
cd "${REPO_ROOT}"

ARRAY_COUNT="$(python3 scripts/ppo_benchmark.py --profile "${PROFILE}" --matrix-count)"
ARRAY_END="$((ARRAY_COUNT - 1))"

echo "Submitting ${ARRAY_COUNT} tasks for profile ${PROFILE}"
sbatch --array="0-${ARRAY_END}" \
  --chdir="${REPO_ROOT}" \
  --export=ALL,PROFILE="${PROFILE}",LOG_WANDB="${LOG_WANDB}" \
  scripts/slurm_benchmark_ppo_vs_contrastive.sh
