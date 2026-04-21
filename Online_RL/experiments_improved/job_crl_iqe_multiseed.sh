#!/bin/bash
# =============================================================================
#  SLURM ARRAY job — 2 variants × 5 seeds = 10 independent tasks
#
#  Runs CRL baseline and CRL + IQE on Ant Soccer, seeds 1..5.
#  Each array task is ONE (variant, seed) run → fits comfortably in 3 h.
#
#  Submit with:
#      sbatch job_crl_iqe_multiseed.sh
#
#  Monitor with:
#      squeue -u $USER
#      tail -f logs/crl_multiseed_<JOBID>_<TASKID>.out
#
#  Cancel all tasks:
#      scancel <JOBID>
# =============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=crl_iqe_multiseed
#SBATCH --account=rob530w26s001_class
# NOTE: no --chdir here on purpose — sbatch uses its launch directory, and
# the script cd's to its own location below, so this works from any checkout.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1

# Per-task time limit — one 50M-step run on Ant Soccer ≈ 90 min.
#SBATCH --time=03:00:00

# Array: 10 tasks (0..9). Mapping below: TASK / 5 = variant idx, TASK % 5 = seed idx.
#SBATCH --array=0-9

# Per-task stdout/err so each run writes its own log file.
#SBATCH --output=logs/crl_multiseed_%A_%a.out
#SBATCH --error=logs/crl_multiseed_%A_%a.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shayesteh7813@gmail.com

# =============================================================================
# Setup
# =============================================================================
set -euo pipefail

# Auto-detect the directory this script lives in — portable across checkouts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  CRL+IQE Multi-seed Array Task"
echo "  Job ID        : ${SLURM_JOB_ID}"
echo "  Array Job ID  : ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "  Array Task ID : ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "  Node          : $(hostname)"
echo "  Start         : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── GPU info ─────────────────────────────────────────────────────────────────
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not found)"
echo ""

# ── Activate conda environment ───────────────────────────────────────────────
module purge
module load cuda/12.3.0

echo "Activating conda environment: jaxgcrl"
eval "$(conda shell.bash hook)"
conda activate jaxgcrl

echo "Wiping and reinstalling numpy (avoid mixed 1.x/2.x residue)..."
SITE_PKGS_TMP="${CONDA_PREFIX}/lib/python3.10/site-packages"
pip uninstall numpy -y --quiet 2>/dev/null || true
rm -rf "${SITE_PKGS_TMP}/numpy" "${SITE_PKGS_TMP}/numpy"-*.dist-info 2>/dev/null || true
unset SITE_PKGS_TMP

echo "Restoring packages per environment.yml..."
pip install --quiet \
    "numpy==1.26.4" \
    "scipy==1.12.0" \
    "nvidia-cufft-cu12==11.2.0.44" \
    "nvidia-cusolver-cu12==11.7.3.90" \
    "nvidia-cuda-cupti-cu12==12.4.99"

SITE_PKGS="${CONDA_PREFIX}/lib/python3.10/site-packages"
_LDPATH="${CONDA_PREFIX}/lib"
for _d in "${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "${_d}" ] && _LDPATH="${_d}:${_LDPATH}"
done
export LD_LIBRARY_PATH="${_LDPATH}:${LD_LIBRARY_PATH:-}"
unset _LDPATH _d

echo "JAX devices check:"
python -c "import jax; print('  ', jax.devices())"
echo ""

# ── Environment variables ────────────────────────────────────────────────────
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# Map SLURM_ARRAY_TASK_ID → (variant, seed)
# =============================================================================

RUNNER="${SCRIPT_DIR}/runner_improved.py"

VARIANTS=("baseline" "iqe")
SEEDS=(1 2 3 4 5)

NUM_SEEDS=${#SEEDS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}

VARIANT_IDX=$(( TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( TASK_ID % NUM_SEEDS ))

VARIANT="${VARIANTS[${VARIANT_IDX}]}"
SEED="${SEEDS[${SEED_IDX}]}"

ENV="ant_ball"
TOTAL_STEPS=50000000
NUM_ENVS=1024
NUM_EVALS=200

echo "------------------------------------------------------------"
echo "  Task ${TASK_ID}: variant=${VARIANT}  seed=${SEED}"
echo "  env=${ENV}  total_steps=${TOTAL_STEPS}  num_envs=${NUM_ENVS}"
echo "------------------------------------------------------------"

python "${RUNNER}" \
    --variant "${VARIANT}" \
    --env "${ENV}" \
    --seed "${SEED}" \
    --total_env_steps "${TOTAL_STEPS}" \
    --num_envs "${NUM_ENVS}" \
    --num_evals "${NUM_EVALS}"

echo ""
echo "============================================================"
echo "  Task ${TASK_ID} (variant=${VARIANT}, seed=${SEED}) finished"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# =============================================================================
# If this is the last task in the array, run the aggregated plot
# (NOTE: race-safe only if tasks finish in order. For a robust post-hoc plot
#  after ALL tasks complete, just run `python plot_improved.py` manually
#  or submit a dependent job with `--dependency=afterok:${SLURM_ARRAY_JOB_ID}`.)
# =============================================================================
LAST_TASK=$(( (${#VARIANTS[@]} * NUM_SEEDS) - 1 ))
if [ "${TASK_ID}" -eq "${LAST_TASK}" ]; then
    echo ""
    echo "Last task — generating multi-seed comparison plot..."
    python "${SCRIPT_DIR}/plot_improved.py" --env "${ENV}"
fi
