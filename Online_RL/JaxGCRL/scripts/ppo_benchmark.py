#!/usr/bin/env python3

import argparse
import itertools
import shlex
import subprocess
from pathlib import Path


FULL_ENVS = (
    "reacher",
    "pusher_hard",
    "humanoid",
    "ant",
    "ant_u_maze",
    "ant_big_maze",
    "ant_ball",
    "ant_push",
)
AGENTS = ("ppo", "ppo_contrastive")
PROFILES = {
    "smoke": {
        "envs": ("reacher", "ant"),
        "seeds": (0,),
        "total_env_steps": 1_000_000,
        "num_evals": 10,
    },
    "pilot": {
        "envs": FULL_ENVS,
        "seeds": (0,),
        "total_env_steps": 10_000_000,
        "num_evals": 50,
    },
    "final": {
        "envs": FULL_ENVS,
        "seeds": (0, 1, 2, 3, 4),
        "total_env_steps": 50_000_000,
        "num_evals": 200,
    },
}


def build_command(profile: str, agent: str, env: str, seed: int, log_wandb: bool) -> list[str]:
    profile_config = PROFILES[profile]
    exp_name = f"{profile}_{agent}_{env}"
    command = [
        "python",
        "run.py",
        agent,
        "--env",
        env,
        "--seed",
        str(seed),
        "--total_env_steps",
        str(profile_config["total_env_steps"]),
        "--num_evals",
        str(profile_config["num_evals"]),
        "--exp_name",
        exp_name,
        "--wandb_group",
        f"ppo_vs_contrastive_{profile}",
        "--wandb_project_name",
        "ppo_vs_contrastive",
    ]
    command.append("--log-wandb" if log_wandb else "--no-log-wandb")
    return command


def matrix_entries(profile: str) -> list[tuple[str, str, int]]:
    profile_config = PROFILES[profile]
    return list(itertools.product(AGENTS, profile_config["envs"], profile_config["seeds"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark helper for PPO vs PPO-contrastive.")
    parser.add_argument("--profile", choices=tuple(PROFILES.keys()), required=True)
    parser.add_argument("--agent", choices=AGENTS)
    parser.add_argument("--env", choices=FULL_ENVS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--matrix-count", action="store_true")
    parser.add_argument("--matrix-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.matrix_count:
        print(len(matrix_entries(args.profile)))
        return

    if args.matrix_index is not None:
        entries = matrix_entries(args.profile)
        if args.matrix_index < 0 or args.matrix_index >= len(entries):
            raise ValueError(
                f"Matrix index {args.matrix_index} is out of range for profile {args.profile}."
            )
        agent, env, seed = entries[args.matrix_index]
    else:
        if args.agent is None or args.env is None or args.seed is None:
            raise ValueError("--agent, --env, and --seed are required unless --matrix-index is used.")
        agent, env, seed = args.agent, args.env, args.seed

    command = build_command(args.profile, agent, env, seed, args.log_wandb)
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
