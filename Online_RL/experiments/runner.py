#!/usr/bin/env python3
"""
runner.py — Single-experiment runner for JaxGCRL baselines.

Runs ONE (algo, env, seed) combination, saves metrics to CSV and a JSON
summary, and prints timestamped progress throughout training.

Usage:
    python runner.py --algo crl  --env ant      --seed 1
    python runner.py --algo sac  --env ant_ball --seed 2
    python runner.py --algo ppo  --env ant      --seed 1
    python runner.py --algo td3  --env ant_ball --seed 3

All results land in  experiments/results/<env>/<algo_label>_s<seed>.csv
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths (set BEFORE any JAX/jaxgcrl import) ────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.absolute()
JAXGCRL_DIR = SCRIPT_DIR.parent / "JaxGCRL"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(JAXGCRL_DIR))

# ── MuJoCo / XLA environment variables ───────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".95")
os.environ.setdefault("MUJOCO_GL", "egl")          # headless rendering on HPC

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s|  %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single GCRL baseline (algo × env × seed)."
    )
    p.add_argument(
        "--algo", required=True, choices=["crl", "sac", "ppo", "td3"],
        help="Algorithm: crl | sac (with HER) | ppo | td3 (with HER)",
    )
    p.add_argument(
        "--env", required=True,
        choices=[
            "ant", "ant_random_start", "ant_ball", "ant_push",
            "ant_u_maze", "ant_big_maze", "ant_hardest_maze", "ant_ball_maze",
            "humanoid", "humanoid_u_maze", "humanoid_big_maze", "humanoid_hardest_maze",
            "cheetah", "reacher",
            "pusher_easy", "pusher_hard", "pusher_reacher", "pusher2",
            "arm_reach", "arm_grasp",
            "arm_push_easy", "arm_push_hard",
            "arm_binpick_easy", "arm_binpick_hard",
            "simple_u_maze", "simple_big_maze", "simple_hardest_maze",
        ],
        help="Environment name (see JaxGCRL for details)",
    )
    p.add_argument("--seed",            type=int, default=1)
    p.add_argument("--total_env_steps", type=int, default=50_000_000)
    p.add_argument("--num_envs",        type=int, default=0,
                   help="Parallel envs (0 = auto from paper: PPO→4096, "
                        "humanoid→512, others→1024).")
    p.add_argument("--num_evals",       type=int, default=200,
                   help="Evaluation checkpoints during training.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Agent + RunConfig factories
# ─────────────────────────────────────────────────────────────────────────────

def make_agent(algo: str):
    """Return an agent instance with paper-matching hyperparameters.

    Reference: JaxGCRL paper (Bortkiewicz et al., ICLR 2025)
      - Table 2 (Appendix B.3): CRL benchmark parameters.
      - Section B.2: "discount factor of 0.99 for all methods except PPO,
        for which we use a discount factor of 0.97."
      - Section B.2: "1024 parallel environments for all methods except for
        PPO, where we use 4096 parallel environments."
      - Section 5.1: SAC+HER and TD3+HER use final-strategy relabelling.
    """
    from jaxgcrl.agents.crl import CRL
    from jaxgcrl.agents.ppo import PPO
    from jaxgcrl.agents.sac import SAC
    from jaxgcrl.agents.td3 import TD3

    if algo == "crl":
        # Paper Table 2: policy_lr=6e-4, critic_lr=3e-4,
        # contrastive_loss=sym_infonce, energy=L2 (= code "norm"),
        # discounting=0.99, logsumexp_penalty=0.1
        return CRL(
            policy_lr=6e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            batch_size=256,
            discounting=0.99,
            train_step_multiplier=1,
            max_replay_size=10_000,
            min_replay_size=1_000,
            unroll_length=62,
            contrastive_loss_fn="sym_infonce",
            energy_fn="norm",
            logsumexp_penalty_coeff=0.1,
            h_dim=256,
            n_hidden=2,
            repr_dim=64,
        )

    elif algo == "sac":
        # Paper Section B.2: discounting=0.99 for all methods except PPO.
        # SAC + HER with final-strategy relabelling (Section 5.1).
        return SAC(
            learning_rate=1e-4,
            discounting=0.99,
            batch_size=256,
            train_step_multiplier=1,
            max_replay_size=10_000,
            min_replay_size=1000,
            unroll_length=62,
            use_her=True,
            h_dim=256,
            n_hidden=2,
            tau=0.005,
        )

    elif algo == "ppo":
        # Paper Section B.2: discounting=0.97 for PPO, 4096 parallel envs.
        # Constraint: batch_size * num_minibatches % num_envs == 0
        # With num_envs=4096: 256 * 16 = 4096 ✓
        return PPO(
            learning_rate=3e-4,
            entropy_cost=1e-4,
            discounting=0.97,
            unroll_length=10,
            batch_size=256,
            num_minibatches=16,
            num_updates_per_batch=2,
            normalize_observations=True,
            reward_scaling=10.0,
            clipping_epsilon=0.3,
            gae_lambda=0.95,
            deterministic_eval=False,
            normalize_advantage=True,
        )

    elif algo == "td3":
        # Paper Section B.2: discounting=0.99 for all methods except PPO.
        # TD3 + HER with final-strategy relabelling (Section 5.1).
        return TD3(
            learning_rate=3e-4,
            discounting=0.99,
            batch_size=256,
            train_step_multiplier=1,
            max_replay_size=10_000,
            min_replay_size=1000,
            unroll_length=62,
            use_her=True,
            tau=0.005,
            policy_delay=2,
            exploration_noise=0.4,
            smoothing_noise=0.2,
            noise_clip=0.5,
        )

    raise ValueError(f"Unknown algo: {algo!r}")


def _paper_num_envs(algo: str, env: str) -> int:
    """Return the number of parallel envs matching the paper (Section B.2, Table 2).

    Paper:
      - PPO: 4096 parallel environments.
      - All other methods: 1024, except humanoid variants which use 512.
    """
    if algo == "ppo":
        return 4096

    # Table 2 footnote: "512 for humanoid"
    # NOTE: 512 causes OOM on A100-40GB for humanoid (obs_dim=292 → replay
    # buffer alone is 5.6 GiB).  Use 256 to fit in 40 GiB GPU memory.
    humanoid_envs = {
        "humanoid", "humanoid_u_maze", "humanoid_big_maze", "humanoid_hardest_maze",
    }
    ant_envs = {
        "ant_push"}
    if env in humanoid_envs:
        return 256
    if env in ant_envs:
        return 256

    return 1024


def make_run_config(
    algo: str,
    env: str,
    seed: int,
    total_env_steps: int,
    num_envs: int,
    num_evals: int,
):
    """Return a RunConfig dataclass instance."""
    from jaxgcrl.utils.config import RunConfig

    # Use paper-matching num_envs unless explicitly overridden via CLI
    _num_envs = num_envs
    _episode_length = 1000

    # Reduce eval envs for humanoid to save GPU memory
    humanoid_envs = {
        "humanoid", "humanoid_u_maze", "humanoid_big_maze", "humanoid_hardest_maze",
    }
    _num_eval_envs = 128 if env in humanoid_envs else 256

    # CRL check: num_envs * (episode_length - 1) % batch_size == 0
    # 1024 * 999 = 1,022,976; 1,022,976 % 256 = 0  ✓
    # 512  * 999 = 511,488;   511,488   % 256 = 0  ✓  (humanoid)

    # CRL computes:
    #   num_training_steps_per_epoch = (total_env_steps - num_prefill_env_steps)
    #                                  // (num_evals * env_steps_per_actor_step)
    # and asserts actual_steps >= config.total_env_steps after training.
    # The floor division loses steps, so we must choose total_env_steps such that
    # (total_env_steps - prefill) is exactly divisible by (num_evals * actor_chunk).
    # Formula: total = prefill + ceil((requested - prefill) / D) * D
    _total_env_steps = total_env_steps
    if algo == "crl":
        _unroll_length      = 62          # must match make_agent CRL config
        _min_replay_size    = 1_000       # must match make_agent CRL config
        prefill             = _min_replay_size * _num_envs
        actor_chunk         = _num_envs * _unroll_length
        D                   = num_evals * actor_chunk
        k                   = math.ceil((total_env_steps - prefill) / D)
        _total_env_steps    = prefill + k * D

    return RunConfig(
        env=env,
        total_env_steps=_total_env_steps,
        episode_length=_episode_length,
        num_envs=_num_envs,
        num_eval_envs=_num_eval_envs,
        action_repeat=1,
        num_evals=num_evals,
        seed=seed,
        exp_name=f"{algo}_{env}_s{seed}",
        log_wandb=False,           # wandb disabled — we log to CSV ourselves
        wandb_project_name="jaxgcrl_baselines",
        wandb_group=env,
        wandb_mode="offline",
        visualization_interval=99_999,   # effectively disables HTML rendering
        vis_length=1000,
        max_devices_per_host=1,
        cuda=True,
        # CRL only assigns `params` inside `if config.checkpoint_logdir:`.
        # Without this, train_fn raises UnboundLocalError on return.
        # Use a dedicated checkpoints dir so .pkl files don't pollute results/.
        checkpoint_logdir=str(SCRIPT_DIR / "checkpoints" / env),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Progress callback — saves metrics to CSV and prints live progress
# ─────────────────────────────────────────────────────────────────────────────

class CSVProgressCallback:
    """
    Drop-in replacement for MetricsRecorder.progress.

    All agents call:
        progress_fn(num_steps, metrics, make_policy, params, env, do_render=...)
    so we accept *args/**kwargs to stay compatible with every agent signature.

    Metrics are appended to a CSV row-by-row (safe to inspect mid-run).
    """

    METRICS_TO_LOG = [
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_easy",
        "eval/episode_success_hard",
        "eval/episode_reward",
        "eval/episode_dist",
        "training/actor_loss",
        "training/critic_loss",
        "training/entropy",
        "training/sps",
    ]

    def __init__(
        self,
        csv_path: Path,
        algo: str,
        env: str,
        seed: int,
        total_steps: int,
    ):
        self.csv_path    = csv_path
        self.algo        = algo
        self.env         = env
        self.seed        = seed
        self.total_steps = total_steps
        self.start_time  = time.time()   # wall clock from construction
        self.call_count  = 0
        self._header_written = False

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Always start fresh — delete any leftover data from prior failed runs
        if csv_path.exists():
            csv_path.unlink()
        log.info(f"CSV output → {csv_path}")

    # All agents call progress_fn with (num_steps, metrics, make_policy, params, env, ...)
    def __call__(self, num_steps, metrics, *args, **kwargs):
        elapsed_sec = time.time() - self.start_time
        elapsed_min = elapsed_sec / 60.0
        self.call_count += 1

        pct = 100.0 * num_steps / self.total_steps if self.total_steps > 0 else 0.0

        # ── Build CSV row ─────────────────────────────────────────────────────
        row = {
            "wall_time_sec": round(elapsed_sec, 2),
            "wall_time_min": round(elapsed_min, 4),
            "env_steps":     int(num_steps),
            "eval_number":   self.call_count,
        }
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = str(v)

        # Append row; write header only once
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

        # ── Console progress ──────────────────────────────────────────────────
        success     = metrics.get("eval/episode_success",     float("nan"))
        success_any = metrics.get("eval/episode_success_any", float("nan"))
        reward      = metrics.get("eval/episode_reward",      float("nan"))
        sps         = metrics.get("training/sps",             float("nan"))

        tag = f"[{self.algo.upper()} | {self.env} | seed={self.seed}]"

        if self.call_count == 1:
            log.info(f"{tag} JIT compilation done — training loop started.")

        log.info(
            f"{tag} "
            f"step {num_steps:>10,}/{self.total_steps:,} ({pct:5.1f}%) | "
            f"time {elapsed_min:6.1f} min | "
            f"success {success:.3f} | "
            f"success_any {success_any:.3f} | "
            f"reward {reward:8.2f} | "
            f"sps {sps:,.0f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Resolve num_envs: 0 means "use paper defaults" ────────────────────────
    num_envs = args.num_envs if args.num_envs > 0 else _paper_num_envs(args.algo, args.env)

    # ── Print experiment header ───────────────────────────────────────────────
    log.info("=" * 68)
    log.info(f"  GCRL Baseline Experiment")
    log.info(f"  algo         : {args.algo.upper()}"
             + (" + HER" if args.algo in ("sac", "td3") else ""))
    log.info(f"  environment  : {args.env}")
    log.info(f"  seed         : {args.seed}")
    log.info(f"  total steps  : {args.total_env_steps:,}")
    log.info(f"  num envs     : {num_envs}")
    log.info(f"  num evals    : {args.num_evals}")
    log.info(f"  started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 68)

    # ── JAX device info ───────────────────────────────────────────────────────
    import jax
    log.info(f"JAX version  : {jax.__version__}")
    log.info(f"JAX devices  : {jax.devices()}")
    log.info(f"JAX backend  : {jax.default_backend()}")

    # ── Disable wandb (we use CSV instead) ───────────────────────────────────
    import wandb
    wandb.init(mode="disabled")

    # ── Imports from jaxgcrl ──────────────────────────────────────────────────
    from jaxgcrl.utils.env import create_env

    # ── Build agent and config ────────────────────────────────────────────────
    log.info("Building agent and run config...")
    agent      = make_agent(args.algo)
    run_config = make_run_config(
        algo=args.algo,
        env=args.env,
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        num_envs=num_envs,
        num_evals=args.num_evals,
    )

    # ── Create environments ───────────────────────────────────────────────────
    log.info(f"Creating environments ({args.env})...")
    train_env = create_env(args.env)
    eval_env  = create_env(args.env)

    # ── CSV output path ───────────────────────────────────────────────────────
    # SAC and TD3 use HER, so label them accordingly
    algo_label = (
        f"{args.algo}_her" if args.algo in ("sac", "td3") else args.algo
    )
    csv_path = RESULTS_DIR / args.env / f"{algo_label}_s{args.seed}.csv"

    progress_fn = CSVProgressCallback(
        csv_path    = csv_path,
        algo        = args.algo,
        env         = args.env,
        seed        = args.seed,
        total_steps = args.total_env_steps,
    )

    # ── Ensure checkpoint directory exists ────────────────────────────────────
    # CRL uses checkpoint_logdir as a directory (saves step_N.pkl inside it).
    # SAC/TD3 use it as a path prefix (saves {logdir}_sac_N.pkl), so the
    # parent directory must exist.  Creating the full path satisfies both.
    Path(run_config.checkpoint_logdir).mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────
    log.info("Starting training (first call includes JIT compilation)...")
    t0 = time.time()

    try:
        _, params, _ = agent.train_fn(
            train_env   = train_env,
            eval_env    = eval_env,
            config      = run_config,
            progress_fn = progress_fn,
        )
    except Exception as exc:
        log.error(f"Training failed with exception: {exc}")
        raise

    wall_min = (time.time() - t0) / 60.0
    log.info(f"Training finished in {wall_min:.1f} minutes.")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / args.env / f"{algo_label}_s{args.seed}_summary.json"
    summary = {
        "algo":             args.algo,
        "algo_label":       algo_label,
        "env":              args.env,
        "seed":             args.seed,
        "total_env_steps":  run_config.total_env_steps,
        "num_envs":         num_envs,
        "num_evals":        args.num_evals,
        "wall_time_min":    round(wall_min, 2),
        "eval_calls":       progress_fn.call_count,
        "csv_path":         str(csv_path),
        "completed_at":     datetime.now().isoformat(),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 68)
    log.info(f"  Metrics CSV  → {csv_path}")
    log.info(f"  Summary JSON → {summary_path}")
    log.info(f"  Wall time    : {wall_min:.1f} min")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
