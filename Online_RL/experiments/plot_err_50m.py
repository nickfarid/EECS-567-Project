#!/usr/bin/env python3
"""
plot_err_50m.py — Paper-style plots of success_any from 50M .err log files.

Parses per-algorithm .err files, plots 11 equally-spaced markers per curve.
Missing algorithms are shown as flat zero lines.  Generates individual figures
per environment plus a 2x4 grid figure matching the paper layout.

Usage:
    python plot_err_50m.py
    python plot_err_50m.py --n_points 11
"""

import argparse
import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s|  %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.absolute()
LOGS_DIR    = SCRIPT_DIR / "logs"
FIGURES_DIR = SCRIPT_DIR / "figures"

# ── Display config ────────────────────────────────────────────────────────────
ALGO_CFG: Dict[str, dict] = {
    "crl":     {"label": "CRL",       "color": "#3171AD", "marker": "o"},
    "sac_her": {"label": "SAC + HER", "color": "#469C76", "marker": "s"},
    "ppo":     {"label": "PPO",       "color": "#C66526", "marker": "^"},
    "td3_her": {"label": "TD3 + HER", "color": "#C19368", "marker": "D"},
}

ENV_DISPLAY = {
    "reacher":      "Reacher",
    "pusher_hard":  "Pusher Hard",
    "humanoid":     "Humanoid",
    "ant":          "Ant",
    "ant_u_maze":   "Ant U-Maze",
    "ant_big_maze": "Ant Big Maze",
    "ant_ball":     "Ant Soccer",
    "ant_push":     "Ant Push",
}

# Environment order for the 2x4 grid (row-major)
ENV_ORDER = [
    "reacher", "pusher_hard", "humanoid", "ant",
    "ant_u_maze", "ant_big_maze", "ant_ball", "ant_push",
]

# Map log bracket algo names -> ALGO_CFG keys
LOG_ALGO_MAP = {"CRL": "crl", "SAC": "sac_her", "PPO": "ppo", "TD3": "td3_her"}

# ── File-to-environment mapping ──────────────────────────────────────────────
# Each entry: (filename, env_key, set of algo_keys to keep from that file)
# None means keep all algos found in the file.
FILE_CFG: List[Tuple[str, str, Optional[set]]] = [
    # Reacher (10M) — CRL in one file, SAC/PPO/TD3 in another
    ("reacher_crl.err",            "reacher",      {"crl"}),
    ("reacher_sac_td3_ppo.err",    "reacher",      {"sac_her", "ppo", "td3_her"}),
    # Pusher Hard
    ("pusher_hard_crl.err",        "pusher_hard",  {"crl"}),
    ("pusher_hard_td3.err",        "pusher_hard",  {"td3_her"}),
    # Humanoid
    ("humanoid_crl.err",           "humanoid",     {"crl"}),
    # Ant
    ("ant_crl.err",                "ant",          {"crl"}),
    ("ant_sac.err",                "ant",          {"sac_her"}),
    # Ant U-Maze
    ("ant_u_maze_crl.err",         "ant_u_maze",   {"crl"}),
    ("ant_u_maze_sac.err",         "ant_u_maze",   {"sac_her"}),
    # Ant Big Maze
    ("ant_big_maze_crl.err",       "ant_big_maze", {"crl"}),
    ("ant_big_maze_sac.err",       "ant_big_maze", {"sac_her"}),
    # Ant Soccer (env logged as ant_ball)
    ("ant_soccer_crl.err",         "ant_ball",     {"crl"}),
    # Ant Push
    ("ant_push_crl.err",           "ant_push",     {"crl"}),
]

# Regex to extract algo, time, and success_any from log lines
LINE_RE = re.compile(
    r'\[(\w+)\s*\|.*?'
    r'time\s+([\d.]+)\s*min.*?'
    r'success_any\s+([\d.]+)'
)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_err_file(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Parse a .err log file and return per-algo lists of (time_min, success_any)."""
    data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m is None:
                continue
            algo_key = LOG_ALGO_MAP.get(m.group(1))
            if algo_key is None:
                continue
            t = float(m.group(2))
            val = float(m.group(3))
            data[algo_key].append((t, val))
    return dict(data)


def build_env_data() -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Load all .err files and assemble per-env, per-algo data."""
    env_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(dict)

    for filename, env, keep_algos in FILE_CFG:
        path = LOGS_DIR / filename
        if not path.exists():
            log.warning(f"File not found: {path}")
            continue

        parsed = parse_err_file(path)
        log.info(f"Parsed {filename}: "
                 f"{', '.join(f'{k}({len(v)} pts)' for k, v in parsed.items())}")

        for algo_key, series in parsed.items():
            if keep_algos is not None and algo_key not in keep_algos:
                continue
            env_data[env][algo_key] = series

    return dict(env_data)


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_to_grid(
    series: List[Tuple[float, float]],
    n_points: int,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate a (time, value) series to n_points equally-spaced points."""
    t = np.array([p[0] for p in series])
    y = np.array([p[1] for p in series])
    order = np.argsort(t)
    t, y = t[order], y[order]
    y = np.clip(y, 0.0, 1.0)
    t_grid = np.linspace(0, t_end, n_points)
    y_grid = np.interp(t_grid, t, y)
    return t_grid, y_grid


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7,
    "lines.linewidth":  1.8,
    "lines.markersize": 5,
    "figure.dpi":       150,
})


def _style_ax(ax: plt.Axes, t_max: float) -> None:
    """Apply common axis styling."""
    ax.set_xlabel("Time in minutes")
    ax.set_ylabel("Success rate")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in")


def plot_env_on_ax(
    ax: plt.Axes,
    env: str,
    algo_data: Dict[str, List[Tuple[float, float]]],
    n_points: int = 11,
) -> None:
    """Plot all 4 algorithms for one environment on the given axes."""

    # Find max time across available algorithms
    t_max = 0.0
    algo_end_times: Dict[str, float] = {}
    for algo_key in ALGO_CFG:
        if algo_key in algo_data and algo_data[algo_key]:
            end_t = max(p[0] for p in algo_data[algo_key])
            algo_end_times[algo_key] = end_t
            t_max = max(t_max, end_t)

    if t_max == 0.0:
        t_max = 1.0  # fallback

    # Plot each algorithm
    for algo_key, cfg in ALGO_CFG.items():
        if algo_key in algo_data and algo_data[algo_key]:
            # Available algorithm — interpolate to its own end time
            end_t = algo_end_times[algo_key]
            t_grid, y_grid = interpolate_to_grid(
                algo_data[algo_key], n_points, end_t
            )
        else:
            # Missing algorithm — flat zero line to t_max
            # Hard-coded to t_max so user can change it per env
            t_grid = np.linspace(0, t_max, n_points)
            y_grid = np.zeros(n_points)

        ax.plot(
            t_grid, y_grid,
            color=cfg["color"],
            linestyle="-",
            marker=cfg["marker"],
            markersize=5,
            label=cfg["label"],
        )

    _style_ax(ax, t_max)
    ax.set_title(ENV_DISPLAY.get(env, env), fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)


def save_fig(fig: plt.Figure, stem: str) -> None:
    """Save figure as PNG and PDF."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        out = FIGURES_DIR / f"{stem}.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        log.info(f"  Saved -> {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure generators
# ─────────────────────────────────────────────────────────────────────────────

def fig_per_env(env_data: dict, n_points: int) -> None:
    """One figure per environment."""
    for env in ENV_ORDER:
        display = ENV_DISPLAY.get(env, env)
        log.info(f"\n[Figure] {display}")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_env_on_ax(ax, env, env_data.get(env, {}), n_points=n_points)
        fig.tight_layout()
        save_fig(fig, f"{env}_50m_comparison")


def fig_grid(env_data: dict, n_points: int) -> None:
    """2x4 grid figure matching the paper layout."""
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for idx, env in enumerate(ENV_ORDER):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        display = ENV_DISPLAY.get(env, env)
        log.info(f"  [{r},{c}] {display}")
        plot_env_on_ax(ax, env, env_data.get(env, {}), n_points=n_points)

    fig.tight_layout()
    save_fig(fig, "baselines_50m_grid")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Paper-style plots from 50M .err log files"
    )
    p.add_argument(
        "--n_points", type=int, default=11,
        help="Number of equally-spaced markers per curve (default: 11)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    n_points = args.n_points

    log.info("=" * 60)
    log.info("  Paper-style plots from 50M .err log files")
    log.info(f"  Points/curve  : {n_points}")
    log.info(f"  Logs dir      : {LOGS_DIR}")
    log.info(f"  Figures dir   : {FIGURES_DIR}")
    log.info("=" * 60)

    env_data = build_env_data()

    # Individual per-environment figures
    fig_per_env(env_data, n_points)

    # 2x4 grid figure
    log.info("\n[Grid figure]")
    fig_grid(env_data, n_points)

    log.info("\nAll done!")


if __name__ == "__main__":
    main()
