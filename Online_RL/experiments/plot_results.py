#!/usr/bin/env python3
"""
plot_results.py — Visualise GCRL baseline results.

Reads CSV files produced by runner.py and generates:
  1. baselines_comparison.{png,pdf}  — side-by-side Ant / Ant Soccer (paper style)
  2. ant_comparison.{png,pdf}        — Ant only
  3. ant_ball_comparison.{png,pdf}   — Ant Soccer only
  4. per_algo_grid.{png,pdf}         — one column per algorithm, two env lines each
  5. A printed summary table of final success rates

All figures are saved to  experiments/figures/

Usage:
    python plot_results.py
    python plot_results.py --t_max 35      # clip x-axis at 35 minutes
    python plot_results.py --metric eval/episode_success_any
"""

import argparse
import csv as _csv_module
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on HPC nodes
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
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

# ── Display config ────────────────────────────────────────────────────────────
ALGO_CFG: Dict[str, dict] = {
    "crl":     {"label": "CRL",       "color": "#1f77b4", "ls": "-",  "marker": "o"},
    "sac_her": {"label": "SAC + HER", "color": "#2ca02c", "ls": "-",  "marker": "s"},
    "ppo":     {"label": "PPO",       "color": "#ff7f0e", "ls": "-",  "marker": "^"},
    "td3_her": {"label": "TD3 + HER", "color": "#d62728", "ls": "-",  "marker": "D"},
}

ENV_DISPLAY = {
    "ant":                  "Ant",
    "ant_random_start":     "Ant (Random Start)",
    "ant_ball":             "Ant Soccer",
    "ant_push":             "Ant Push",
    "ant_ball_maze":        "Ant Ball Maze",
    "ant_u_maze":           "Ant U-Maze",
    "ant_big_maze":         "Ant Big Maze",
    "ant_hardest_maze":     "Ant Hardest Maze",
    "humanoid":             "Humanoid",
    "humanoid_u_maze":      "Humanoid U-Maze",
    "humanoid_big_maze":    "Humanoid Big Maze",
    "humanoid_hardest_maze":"Humanoid Hardest Maze",
    "cheetah":              "Cheetah",
    "reacher":              "Reacher",
    "pusher_easy":          "Pusher (Easy)",
    "pusher_hard":          "Pusher (Hard)",
    "pusher_reacher":       "Pusher Reacher",
    "pusher2":              "Pusher 2",
    "arm_reach":            "Arm Reach",
    "arm_grasp":            "Arm Grasp",
    "arm_push_easy":        "Arm Push (Easy)",
    "arm_push_hard":        "Arm Push (Hard)",
    "arm_binpick_easy":     "Arm Bin Pick (Easy)",
    "arm_binpick_hard":     "Arm Bin Pick (Hard)",
    "simple_u_maze":        "Simple U-Maze",
    "simple_big_maze":      "Simple Big Maze",
    "simple_hardest_maze":  "Simple Hardest Maze",
}

DEFAULT_METRIC = "eval/episode_success"
DEFAULT_T_MAX  = 35.0   # minutes — matches the paper's x-axis


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> Optional[Dict[str, list]]:
    """Load a CSV file into a dict of column → list of values.

    Handles CSVs that contain multiple header rows (from appended runs) by
    keeping only the rows after the *last* occurrence of the header.
    """
    try:
        # Find the last header line so we only use the most recent run's data
        text = path.read_text()
        lines = text.splitlines()
        if not lines:
            return None
        header = lines[0]
        # Find the last occurrence of the header row
        last_header_idx = 0
        for i, line in enumerate(lines):
            if line == header:
                last_header_idx = i
        # Parse only rows after the last header
        relevant_lines = [lines[last_header_idx]] + [
            l for l in lines[last_header_idx + 1:] if l.strip() and l != header
        ]
        from io import StringIO as _StringIO
        reader = _csv_module.DictReader(_StringIO("\n".join(relevant_lines)))
        data: Dict[str, list] = {}
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    pass  # skip non-numeric values
        return data if data else None
    except Exception as e:
        log.warning(f"  csv load failed on {path.name}: {e}")
        return None


def load_runs(env: str, algo: str) -> List[Dict[str, np.ndarray]]:
    """
    Return a list of dicts (one per seed) mapping column name → np.ndarray.
    Files are expected at  results/<env>/<algo>_s*.csv
    """
    env_dir = RESULTS_DIR / env
    if not env_dir.exists():
        log.warning(f"Results dir not found: {env_dir}")
        return []

    files = sorted(env_dir.glob(f"{algo}_s*.csv"))
    if not files:
        log.info(f"  No CSV files for {env}/{algo}")
        return []

    runs = []
    for f in files:
        data = _load_csv(f)
        if data is None:
            continue
        required = {"wall_time_min", "env_steps"}
        if not required.issubset(data.keys()):
            log.warning(f"  {f.name}: missing required columns {required - set(data.keys())}")
            continue
        # Convert all numeric columns to numpy arrays
        np_data = {}
        for k, v in data.items():
            try:
                np_data[k] = np.array(v, dtype=float)
            except (ValueError, TypeError):
                np_data[k] = v   # keep as-is for string columns
        runs.append(np_data)
        log.info(f"  Loaded {f.name}: {len(next(iter(np_data.values())))} eval points")

    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────

def smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple symmetric moving-average; preserves array length."""
    if len(y) < window or window <= 1:
        return y
    kernel = np.ones(window) / window
    # 'same' convolution keeps the same length; edge artefacts are minor for
    # window=3 and long sequences.
    return np.convolve(y, kernel, mode="same")


def interpolate_to_common_grid(
    runs:     List[Dict[str, np.ndarray]],
    metric:   str,
    t_col:    str  = "wall_time_min",
    n_points: int  = 300,
    t_max:    float = DEFAULT_T_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate all seeds to a shared time grid.

    Returns:
        t_grid  : (n_points,) time axis in minutes
        mean    : (n_points,) mean across seeds
        std     : (n_points,) std  across seeds
    """
    valid_runs = [r for r in runs if metric in r and t_col in r]
    if not valid_runs:
        # Return zeros if metric is missing (e.g. PPO doesn't have training/alpha_loss)
        t_grid = np.linspace(0, t_max, n_points)
        return t_grid, np.zeros(n_points), np.zeros(n_points)

    # Use the minimum observed t_max across seeds so we only interpolate where
    # all seeds have data.
    t_max_data = min(float(r[t_col].max()) for r in valid_runs)
    t_max_use  = min(t_max, t_max_data)
    t_grid = np.linspace(0, t_max_use, n_points)

    interped = []
    for r in valid_runs:
        t = r[t_col].astype(float)
        y = r[metric].astype(float)
        # Sort by time (should already be sorted, but just in case)
        order = np.argsort(t)
        t, y = t[order], y[order]
        # Clamp to non-negative success rates
        y = np.clip(y, 0.0, 1.0) if "success" in metric else y
        interped.append(np.interp(t_grid, t, y))

    arr  = np.array(interped)          # shape: (n_seeds, n_points)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
    return t_grid, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "lines.linewidth":  2.0,
    "figure.dpi":       120,
})


def _style_ax(ax: plt.Axes, t_max: float, metric: str) -> None:
    """Apply common axis styling."""
    ax.set_xlabel("Time in minutes")
    ax.set_ylabel("Success rate" if "success" in metric else metric.split("/")[-1])
    ax.set_xlim(0, t_max)
    if "success" in metric:
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in")


def plot_env_on_ax(
    ax:     plt.Axes,
    env:    str,
    metric: str,
    t_max:  float,
    smooth_window: int = 3,
) -> bool:
    """
    Draw one line per algorithm onto *ax* for the given environment.
    Returns True if at least one algorithm had data.
    """
    has_data = False

    for algo, cfg in ALGO_CFG.items():
        log.info(f"    loading {env}/{algo} ...")
        runs = load_runs(env, algo)

        if not runs:
            continue

        has_data = True
        t_grid, mean, std = interpolate_to_common_grid(
            runs, metric=metric, t_max=t_max
        )
        mean = smooth(mean, window=smooth_window)
        std  = smooth(std,  window=smooth_window)

        n_seeds = len(runs)
        label   = cfg["label"] + (f" (n={n_seeds})" if n_seeds > 1 else "")

        ax.plot(t_grid, mean, color=cfg["color"], linestyle=cfg["ls"], label=label)

        if n_seeds > 1:
            ax.fill_between(
                t_grid,
                np.clip(mean - std, 0, 1) if "success" in metric else mean - std,
                np.clip(mean + std, 0, 1) if "success" in metric else mean + std,
                color=cfg["color"],
                alpha=0.18,
            )

    _style_ax(ax, t_max, metric)
    ax.set_title(ENV_DISPLAY.get(env, env), fontweight="bold")

    if has_data:
        ax.legend(loc="upper left", framealpha=0.9)

    return has_data


def save_fig(fig: plt.Figure, stem: str, formats: Tuple[str, ...] = ("png", "pdf")) -> None:
    """Save figure in multiple formats and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = FIGURES_DIR / f"{stem}.{fmt}"
        try:
            fig.savefig(out, dpi=150)
            log.info(f"  Saved → {out}")
        except Exception as e:
            log.warning(f"  Could not save {out.name}: {e}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure generators
# ─────────────────────────────────────────────────────────────────────────────

def discover_envs() -> List[str]:
    """Auto-discover environments from results/ subdirectories."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name for d in RESULTS_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.csv"))
    )


def fig_per_env(metric: str, t_max: float) -> None:
    """One figure per environment (auto-discovered from results/)."""
    envs = discover_envs()
    if not envs:
        log.warning("No environment results found in %s", RESULTS_DIR)
        return
    for env in envs:
        display = ENV_DISPLAY.get(env, env)
        log.info(f"\n[Figure] {display}")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_env_on_ax(ax, env=env, metric=metric, t_max=t_max)
        try:
            plt.tight_layout()
        except Exception:
            pass
        save_fig(fig, f"{env}_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(metric: str) -> None:
    """Print a table of final success rates (mean ± std across seeds)."""
    col_w = 16

    print("\n" + "=" * 68)
    print("  FINAL RESULTS SUMMARY")
    print(f"  metric : {metric}")
    print("=" * 68)

    for env in discover_envs():
        display = ENV_DISPLAY.get(env, env)
        print(f"\n  {display}")
        print(f"  {'Algorithm':<{col_w}} {'Seeds':>6}  {'Final success (mean ± std)':>28}")
        print("  " + "-" * 54)

        for algo, cfg in ALGO_CFG.items():
            runs = load_runs(env, algo)
            finals = []
            for r in runs:
                if metric in r and len(r[metric]) > 0:
                    finals.append(float(r[metric][-1]))

            if finals:
                m = np.mean(finals)
                s = np.std(finals)
                print(f"  {cfg['label']:<{col_w}} {len(finals):>6}  {m:.3f} ± {s:.3f}")
            else:
                print(f"  {cfg['label']:<{col_w}} {'—':>6}  {'no data':>28}")

    print("\n" + "=" * 68)


def print_timing_table() -> None:
    """Print wall-clock training times from the JSON summaries."""
    import json, glob

    print("\n" + "=" * 68)
    print("  TRAINING TIMES (wall clock)")
    print("=" * 68)

    for env in discover_envs():
        display = ENV_DISPLAY.get(env, env)
        print(f"\n  {display}")
        for algo, cfg in ALGO_CFG.items():
            pattern = str(RESULTS_DIR / env / f"{algo}_s*_summary.json")
            files = sorted(glob.glob(pattern))
            times = []
            for fp in files:
                with open(fp) as f:
                    d = json.load(f)
                times.append(d.get("wall_time_min", float("nan")))
            if times:
                mean_t = np.mean(times)
                print(f"  {cfg['label']:<16} {mean_t:.1f} min  "
                      f"(seeds: {', '.join(f'{t:.1f}' for t in times)})")
            else:
                print(f"  {cfg['label']:<16} no summary found")

    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot GCRL baseline results")
    p.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric to plot (default: {DEFAULT_METRIC})",
    )
    p.add_argument(
        "--t_max",
        type=float,
        default=DEFAULT_T_MAX,
        help="X-axis limit in minutes (default: 35)",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="Moving-average window for smoothing (default: 3, use 1 to disable)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    metric = args.metric
    t_max  = args.t_max

    log.info("=" * 68)
    log.info(f"  Plotting metric : {metric}")
    log.info(f"  X-axis limit    : {t_max} min")
    log.info(f"  Results dir     : {RESULTS_DIR}")
    log.info(f"  Figures dir     : {FIGURES_DIR}")
    log.info("=" * 68)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Print tables ──────────────────────────────────────────────────────────
    print_summary_table(metric)
    print_timing_table()

    # ── Generate figures (one per environment) ──────────────────────────────
    fig_per_env(metric, t_max)

    # ── Also plot "success_any" if we plotted "success" ───────────────────────
    if metric == "eval/episode_success":
        log.info("\n[Bonus] Also plotting eval/episode_success_any ...")
        fig_per_env("eval/episode_success_any", t_max)

    log.info("\nAll done!")
    log.info(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
