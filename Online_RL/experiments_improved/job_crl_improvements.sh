#!/bin/bash
# =============================================================================
#  SLURM job script — University of Michigan Great Lakes cluster
#  Runs 5 CRL variants on Ant Soccer (ant_ball), seed=1 each:
#      baseline + Idea 1 (temp) + Idea 2 (hardneg) + Idea 3 (fwdyn) + Idea 4 (iqe)
#  Total: 5 sequential runs × ~90 min (50M steps) ≈ 7.5 h
#
#  Submit with:
#      sbatch job_crl_improvements.sh
#
#  Monitor with:
#      squeue -u $USER
#      tail -f logs/crl_improvements.out
# =============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────

#SBATCH --job-name=crl_improvements
#SBATCH --account=rob530w26s001_class
# NOTE: no --chdir here on purpose — sbatch uses its launch directory, and
# the script cd's to its own location below, so this works from any checkout.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1

# 5 runs × ~90 min each (50M steps on Ant Soccer) ≈ 7.5 h → request 10 h
#SBATCH --time=8:00:00

#SBATCH --output=logs/crl_improvements_%j.out
#SBATCH --error=logs/crl_improvements_%j.err

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
echo "  CRL Improvements — Great Lakes SLURM Job"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Node     : $(hostname)"
echo "  Start    : $(date '+%Y-%m-%d %H:%M:%S')"
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

# Clean-wipe numpy to eliminate any mixed 1.x/2.x file residue, then
# reinstall pinned versions from environment.yml.
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

# Build LD_LIBRARY_PATH from pip-installed nvidia lib dirs + conda lib dir.
SITE_PKGS="${CONDA_PREFIX}/lib/python3.10/site-packages"
_LDPATH="${CONDA_PREFIX}/lib"
for _d in "${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "${_d}" ] && _LDPATH="${_d}:${_LDPATH}"
done
export LD_LIBRARY_PATH="${_LDPATH}:${LD_LIBRARY_PATH:-}"
unset _LDPATH _d

echo "JAX devices check:"
python -c "import jax; print('  ', jax.devices())"

echo "Python   : $(which python)"
echo "Python v : $(python --version)"
python -c "import jax; print(f'JAX      : {jax.__version__}')"
python -c "import jax; print(f'Devices  : {jax.devices()}')"
echo ""

# ── Environment variables ────────────────────────────────────────────────────
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# Experiment matrix — 5 CRL variants on Ant Soccer, seed=1
# =============================================================================

RUNNER="${SCRIPT_DIR}/runner_improved.py"

ENV="ant_push"
SEED=1
TOTAL_STEPS=50000000
NUM_ENVS=1024
NUM_EVALS=200

VARIANTS=("baseline" "temp" "hardneg" "fwdyn" "iqe")

TOTAL_RUNS=${#VARIANTS[@]}
RUN_IDX=0

echo "Experiment matrix:"
echo "  Env          : ${ENV}"
echo "  Seed         : ${SEED}"
echo "  Variants     : ${VARIANTS[*]}"
echo "  Total runs   : ${TOTAL_RUNS}"
echo "  Steps/run    : ${TOTAL_STEPS}"
echo "  Num envs     : ${NUM_ENVS}"
echo "  Num evals    : ${NUM_EVALS}"
echo ""

# =============================================================================
# Run all variants sequentially
# =============================================================================

for variant in "${VARIANTS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    RUN_START=$(date '+%Y-%m-%d %H:%M:%S')

    echo "------------------------------------------------------------"
    echo "  Run ${RUN_IDX} / ${TOTAL_RUNS}"
    echo "  variant = ${variant}"
    echo "  env     = ${ENV}"
    echo "  seed    = ${SEED}"
    echo "  start   = ${RUN_START}"
    echo "------------------------------------------------------------"

    # 'set +e' so a single run failure doesn't kill the whole job.
    set +e
    python "${RUNNER}" \
        --variant "${variant}" \
        --env "${ENV}" \
        --seed "${SEED}" \
        --total_env_steps "${TOTAL_STEPS}" \
        --num_envs "${NUM_ENVS}" \
        --num_evals "${NUM_EVALS}"
    RUN_EXIT=$?
    set -e

    if [ ${RUN_EXIT} -ne 0 ]; then
        echo "  WARNING: variant '${variant}' failed (exit ${RUN_EXIT}) — continuing"
    else
        echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo ""
done

# =============================================================================
# Generate comparison figure
# =============================================================================

echo "============================================================"
echo "  All ${TOTAL_RUNS} runs completed!"
echo "  Generating 5-way comparison figure..."
echo "============================================================"
echo ""

python "${SCRIPT_DIR}/plot_improved.py" --env "${ENV}" --seed "${SEED}"

echo ""
echo "============================================================"
echo "  Done!"
echo "  Results  : ${SCRIPT_DIR}/results/${ENV}/"
echo "  Figures  : ${SCRIPT_DIR}/figures/"
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
