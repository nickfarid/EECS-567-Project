#!/bin/bash
# =============================================================================
#  SLURM job script — University of Michigan Great Lakes cluster
#  Runs CRL, SAC+HER, PPO, TD3+HER on Ant and Ant Soccer (2 seeds each)
#  Total: 4 algos × 2 envs × 2 seeds = 16 sequential runs
#
#  Submit with:
#      sbatch job.sh
#
#  Monitor with:
#      squeue -u $USER
#      tail -f logs/job_<JOBID>.out
# =============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────

#SBATCH --job-name=ant_gcrl_baselines
#SBATCH --account=rob530w26s001_class #ece567w26_class
#SBATCH --chdir=/home/shafezi/courses/ECE567/final_project/experiments

# Great Lakes GPU partition — has NVIDIA A100 (40 GB) nodes
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1

# 20 runs × ~20 min each (5 seeds) = ~400 min → fits within the 8 h MaxWall limit
#SBATCH --time=08:00:00

#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Email notifications (optional — remove if not needed)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shayesteh7813@gmail.com

# =============================================================================
# Setup
# =============================================================================
set -euo pipefail

SCRIPT_DIR="/home/shafezi/courses/ECE567/final_project/experiments"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  GCRL Baselines — Great Lakes SLURM Job"
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

# ── Activate conda environment ────────────────────────────────────────────────
module purge
module load cuda/12.3.0

echo "Activating conda environment: jaxgcrl"
eval "$(conda shell.bash hook)"
conda activate jaxgcrl

# Clean-wipe numpy to eliminate any mixed 1.x/2.x file residue, then
# reinstall pinned versions from environment.yml.
# cuDNN comes from conda (cudnn=8.9 in environment.yml), not from pip.
# We do NOT touch nvidia-cublas, nvidia-cuda-runtime, etc. — they are not
# in environment.yml and our prior reinstalls of them caused the breakage.
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

# Build LD_LIBRARY_PATH from every pip-installed nvidia lib dir that exists,
# plus conda's lib dir (contains libcudnn.so.8 from conda cudnn=8.9).
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

# ── Environment variables ─────────────────────────────────────────────────────
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export MUJOCO_GL=egl           # headless OpenGL rendering
export CUDA_VISIBLE_DEVICES=0  # use GPU 0 (SLURM may assign a different index)

# =============================================================================
# Experiment matrix
# =============================================================================

RUNNER="${SCRIPT_DIR}/runner.py"

# Available environments (all supported by JaxGCRL):
#   Locomotion : ant, ant_random_start, humanoid, cheetah, reacher
#   Ball/Push  : ant_ball, ant_push, ant_ball_maze
#   Ant mazes  : ant_u_maze, ant_big_maze, ant_hardest_maze
#   Humanoid m.: humanoid_u_maze, humanoid_big_maze, humanoid_hardest_maze
#   Simple m.  : simple_u_maze, simple_big_maze, simple_hardest_maze
#   Pusher     : pusher_easy, pusher_hard, pusher_reacher, pusher2
#   Arm        : arm_reach, arm_grasp, arm_push_easy, arm_push_hard,
#                arm_binpick_easy, arm_binpick_hard
ENVS=("ant") # "ant_ball")
ALGOS=("crl" "sac" "ppo" "td3")
SEEDS=(1 2 3 4 5)       # 5 seeds → 20 runs → ~6.5 h, safely within the 8 h MaxWall

TOTAL_STEPS=10000000   # 10 M environment steps (matches paper)
NUM_ENVS=512           # parallel environments
NUM_EVALS=50           # evaluation checkpoints during training

# Compute total number of runs for progress display
TOTAL_RUNS=$(( ${#ENVS[@]} * ${#ALGOS[@]} * ${#SEEDS[@]} ))
RUN_IDX=0

echo "Experiment matrix:"
echo "  Environments : ${ENVS[*]}"
echo "  Algorithms   : ${ALGOS[*]}"
echo "  Seeds        : ${SEEDS[*]}"
echo "  Total runs   : ${TOTAL_RUNS}"
echo "  Steps/run    : ${TOTAL_STEPS}"
echo ""

# =============================================================================
# Run all experiments sequentially
# (Sequential is simpler and avoids GPU memory conflicts from parallel runs)
# =============================================================================

for env in "${ENVS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      RUN_IDX=$(( RUN_IDX + 1 ))
      RUN_START=$(date '+%Y-%m-%d %H:%M:%S')

      echo "------------------------------------------------------------"
      echo "  Run ${RUN_IDX} / ${TOTAL_RUNS}"
      echo "  algo  = ${algo}$([ "${algo}" = "sac" ] || [ "${algo}" = "td3" ] && echo " + HER" || echo "")"
      echo "  env   = ${env}"
      echo "  seed  = ${seed}"
      echo "  start = ${RUN_START}"
      echo "------------------------------------------------------------"

      # Use 'set +e' so a single run failure doesn't kill the whole job.
      set +e
      python "${RUNNER}" \
        --algo  "${algo}"        \
        --env   "${env}"         \
        --seed  "${seed}"        \
        --total_env_steps "${TOTAL_STEPS}" \
        --num_envs        "${NUM_ENVS}"    \
        --num_evals       "${NUM_EVALS}"
      RUN_EXIT=$?
      set -e

      if [ ${RUN_EXIT} -ne 0 ]; then
        echo "  WARNING: run ${RUN_IDX} failed (exit ${RUN_EXIT}) — continuing with next run"
      else
        echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
      fi
      echo ""

    done
  done
done

# =============================================================================
# Generate plots and summary
# =============================================================================

echo "============================================================"
echo "  All ${TOTAL_RUNS} runs completed!"
echo "  Generating figures..."
echo "============================================================"
echo ""

python "${SCRIPT_DIR}/plot_results.py"

echo ""
echo "============================================================"
echo "  Done!"
echo "  Results  : ${SCRIPT_DIR}/results/"
echo "  Figures  : ${SCRIPT_DIR}/figures/"
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
