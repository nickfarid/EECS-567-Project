#!/usr/bin/env python3
"""
runner_improved.py — Runner for CRL improvement variants.

Runs ONE (variant, env, seed) combination of **Contrastive RL**, with one of
5 variants:
    baseline  — matches experiments/runner.py CRL exactly
    temp      — Idea 1: learnable temperature τ in InfoNCE
    hardneg   — Idea 2: top-K hardest negatives (K=32 out of 255)
    fwdyn     — Idea 3: forward-dynamics auxiliary loss (coef=0.1)
    iqe       — Idea 4: IQE quasimetric energy function (8 components)

CSV + JSON summary land in
    experiments_improved/results/<env>/crl_<variant>_s<seed>.{csv,json}
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
JAXGCRL_DIR = SCRIPT_DIR.parent / "JaxGCRL_improved"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(JAXGCRL_DIR))

# ── MuJoCo / XLA environment variables ───────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".95")
os.environ.setdefault("MUJOCO_GL", "egl")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s|  %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


VARIANTS = ["baseline", "temp", "hardneg", "fwdyn", "iqe"]

# Environments that need a smaller num_envs to fit in 16–40 GB GPU memory.
# Mirrors the logic in experiments/runner.py `_paper_num_envs`.
MEMORY_HEAVY_ENVS = {
    "ant_push",
    "humanoid",
    "humanoid_u_maze",
    "humanoid_big_maze",
    "humanoid_hardest_maze",
}
DEFAULT_NUM_ENVS_LIGHT = 1024
DEFAULT_NUM_ENVS_HEAVY = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one CRL improvement variant.")
    p.add_argument("--variant", required=True, choices=VARIANTS,
                   help="Which CRL variant to train.")
    p.add_argument("--env", default="ant_ball",
                   help="Environment name (default: ant_ball = Ant Soccer).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total_env_steps", type=int, default=50_000_000)
    p.add_argument("--num_envs", type=int, default=0,
                   help="Parallel envs. 0 = auto (1024, or 256 for heavy envs like "
                        "ant_push/humanoid to fit 16 GB GPUs).")
    p.add_argument("--num_evals", type=int, default=200)
    # Variant-specific knobs (exposed for ablations; defaults match the plan)
    p.add_argument("--hard_neg_top_k", type=int, default=32)
    p.add_argument("--fwd_dyn_coef", type=float, default=0.1)
    p.add_argument("--iqe_num_components", type=int, default=8)
    return p.parse_args()


def make_agent(variant: str, args: argparse.Namespace):
    """Return a CRL agent with baseline hyperparameters plus the requested variant."""
    from jaxgcrl.agents.crl import CRL

    base_kwargs = dict(
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

    if variant == "baseline":
        pass
    elif variant == "temp":
        base_kwargs["use_learnable_temperature"] = True
        base_kwargs["initial_log_tau"] = 0.0
    elif variant == "hardneg":
        base_kwargs["hard_negative_top_k"] = args.hard_neg_top_k
    elif variant == "fwdyn":
        base_kwargs["forward_dynamics_coef"] = args.fwd_dyn_coef
    elif variant == "iqe":
        base_kwargs["energy_fn"] = "iqe"
        base_kwargs["iqe_num_components"] = args.iqe_num_components
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return CRL(**base_kwargs)


def make_run_config(env: str, seed: int, total_env_steps: int,
                    num_envs: int, num_evals: int):
    """Build a RunConfig aligned to CRL's epoch-chunking divisibility constraint."""
    from jaxgcrl.utils.config import RunConfig

    # CRL's internal arithmetic (see JaxGCRL_improved/jaxgcrl/agents/crl/crl.py):
    #   num_prefill_env_steps = min_replay_size * num_envs
    #   env_steps_per_actor_step = num_envs * unroll_length
    #   D = num_evals * env_steps_per_actor_step
    # total_env_steps must satisfy (total - prefill) divisible by D.
    unroll_length   = 62
    min_replay_size = 1_000
    prefill = min_replay_size * num_envs
    actor_chunk = num_envs * unroll_length
    D = num_evals * actor_chunk
    k = math.ceil((total_env_steps - prefill) / D)
    adjusted_total = prefill + k * D

    return RunConfig(
        env=env,
        total_env_steps=adjusted_total,
        episode_length=1000,
        num_envs=num_envs,
        num_eval_envs=256,
        action_repeat=1,
        num_evals=num_evals,
        seed=seed,
        exp_name=f"crl_{env}_s{seed}",
        log_wandb=False,
        wandb_project_name="jaxgcrl_improvements",
        wandb_group=env,
        wandb_mode="offline",
        visualization_interval=99_999,
        vis_length=1000,
        max_devices_per_host=1,
        cuda=True,
        checkpoint_logdir=str(SCRIPT_DIR / "checkpoints" / env),
    )


class CSVProgressCallback:
    """Matches experiments/runner.py CSVProgressCallback schema so that
    plot_improved.py can consume these CSVs identically to the baseline CSVs."""

    def __init__(self, csv_path: Path, variant: str, env: str, seed: int, total_steps: int):
        self.csv_path = csv_path
        self.variant = variant
        self.env = env
        self.seed = seed
        self.total_steps = total_steps
        self.start_time = time.time()
        self.call_count = 0
        self._header_written = False

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            csv_path.unlink()
        log.info(f"CSV output → {csv_path}")

    def __call__(self, num_steps, metrics, *args, **kwargs):
        elapsed_sec = time.time() - self.start_time
        elapsed_min = elapsed_sec / 60.0
        self.call_count += 1
        pct = 100.0 * num_steps / self.total_steps if self.total_steps > 0 else 0.0

        row = {
            "wall_time_sec": round(elapsed_sec, 2),
            "wall_time_min": round(elapsed_min, 4),
            "env_steps": int(num_steps),
            "eval_number": self.call_count,
        }
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = str(v)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

        success = metrics.get("eval/episode_success", float("nan"))
        success_any = metrics.get("eval/episode_success_any", float("nan"))
        reward = metrics.get("eval/episode_reward", float("nan"))
        sps = metrics.get("training/sps", float("nan"))

        tag = f"[CRL-{self.variant.upper()} | {self.env} | seed={self.seed}]"

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


def main():
    args = parse_args()

    # Resolve num_envs: 0 → auto (1024 light / 256 heavy), else honor user value.
    if args.num_envs <= 0:
        args.num_envs = (DEFAULT_NUM_ENVS_HEAVY if args.env in MEMORY_HEAVY_ENVS
                         else DEFAULT_NUM_ENVS_LIGHT)
        log.info(f"num_envs auto-resolved to {args.num_envs} "
                 f"({'heavy env' if args.env in MEMORY_HEAVY_ENVS else 'light env'}).")

    log.info("=" * 68)
    log.info("  CRL Improvement Experiment")
    log.info(f"  variant      : {args.variant}")
    log.info(f"  environment  : {args.env}")
    log.info(f"  seed         : {args.seed}")
    log.info(f"  total steps  : {args.total_env_steps:,}")
    log.info(f"  num envs     : {args.num_envs}")
    log.info(f"  num evals    : {args.num_evals}")
    log.info(f"  started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 68)

    import jax
    log.info(f"JAX version  : {jax.__version__}")
    log.info(f"JAX devices  : {jax.devices()}")
    log.info(f"JAX backend  : {jax.default_backend()}")

    import wandb
    wandb.init(mode="disabled")

    from jaxgcrl.utils.env import create_env

    log.info("Building agent and run config...")
    agent = make_agent(args.variant, args)
    run_config = make_run_config(
        env=args.env,
        seed=args.seed,
        total_env_steps=args.total_env_steps,
        num_envs=args.num_envs,
        num_evals=args.num_evals,
    )

    log.info(f"Creating environments ({args.env})...")
    train_env = create_env(args.env)
    eval_env = create_env(args.env)

    algo_label = f"crl_{args.variant}"
    csv_path = RESULTS_DIR / args.env / f"{algo_label}_s{args.seed}.csv"

    progress_fn = CSVProgressCallback(
        csv_path=csv_path,
        variant=args.variant,
        env=args.env,
        seed=args.seed,
        total_steps=args.total_env_steps,
    )

    Path(run_config.checkpoint_logdir).mkdir(parents=True, exist_ok=True)

    log.info("Starting training (first call includes JIT compilation)...")
    t0 = time.time()

    try:
        _, params, _ = agent.train_fn(
            train_env=train_env,
            eval_env=eval_env,
            config=run_config,
            progress_fn=progress_fn,
        )
    except Exception as exc:
        log.error(f"Training failed with exception: {exc}")
        raise

    wall_min = (time.time() - t0) / 60.0
    log.info(f"Training finished in {wall_min:.1f} minutes.")

    summary_path = RESULTS_DIR / args.env / f"{algo_label}_s{args.seed}_summary.json"
    summary = {
        "algo": "crl",
        "variant": args.variant,
        "algo_label": algo_label,
        "env": args.env,
        "seed": args.seed,
        "total_env_steps": run_config.total_env_steps,
        "num_envs": args.num_envs,
        "num_evals": args.num_evals,
        "wall_time_min": round(wall_min, 2),
        "eval_calls": progress_fn.call_count,
        "csv_path": str(csv_path),
        "completed_at": datetime.now().isoformat(),
        "hyperparams": {
            "hard_neg_top_k": args.hard_neg_top_k if args.variant == "hardneg" else None,
            "fwd_dyn_coef": args.fwd_dyn_coef if args.variant == "fwdyn" else None,
            "iqe_num_components": args.iqe_num_components if args.variant == "iqe" else None,
        },
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
