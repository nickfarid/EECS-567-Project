#!/usr/bin/env python3
"""
plot_improved.py — Compare CRL improvement variants on Ant Soccer.

Auto-aggregates across seeds:
  - 1 seed found: plots a single smoothed curve
  - ≥2 seeds found: plots mean with ±1 std shaded band

Reads CSVs produced by runner_improved.py at:
    experiments_improved/results/<env>/crl_<variant>_s<seed>.csv

Produces:
    experiments_improved/figures/<env>_crl_improvements.{png,pdf}

Usage:
    python plot_improved.py
    python plot_improved.py --env ant_ball
    python plot_improved.py --metric eval/episode_success_any --smooth 3
    python plot_improved.py --variants baseline iqe      # restrict variants shown
"""

import argparse
import csv as _csv_module
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s|  %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

# (variant key, display label, color, linestyle, marker)
VARIANT_CFG: Dict[str, dict] = {
    "baseline": {"label": "CRL (baseline)",    "color": "#333333", "ls": "-",  "marker": "o"},
    "temp":     {"label": "+ learnable τ",      "color": "#1f77b4", "ls": "-",  "marker": "s"},
    "hardneg":  {"label": "+ hard negatives",   "color": "#2ca02c", "ls": "-",  "marker": "^"},
    "fwdyn":    {"label": "+ fwd dynamics aux", "color": "#d62728", "ls": "-",  "marker": "D"},
    "iqe":      {"label": "+ IQE quasimetric",  "color": "#9467bd", "ls": "-",  "marker": "P"},
}

ENV_DISPLAY = {"ant_ball": "Ant Soccer"}


def _load_csv(path: Path) -> Optional[Dict[str, list]]:
    try:
        text = path.read_text()
        lines = text.splitlines()
        if not lines:
            return None
        header_candidate = lines[0]
        header_indices = [i for i, line in enumerate(lines) if line == header_candidate]
        if len(header_indices) > 1:
            lines = lines[header_indices[-1]:]
        rows = list(_csv_module.DictReader(lines))
        if not rows:
            return None
        data: Dict[str, list] = {k: [] for k in rows[0].keys()}
        for row in rows:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (TypeError, ValueError):
                    data[k].append(np.nan)
        return data
    except Exception as exc:
        log.warning(f"Failed to load {path}: {exc}")
        return None


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return out[: len(y)]


def _discover_seed_csvs(env: str, variant: str) -> List[Path]:
    """Find all seed CSVs for a variant: results/<env>/crl_<variant>_s*.csv."""
    env_dir = RESULTS_DIR / env
    if not env_dir.is_dir():
        return []
    return sorted(env_dir.glob(f"crl_{variant}_s*.csv"))


def _interp_curves(seed_data: List[Dict[str, list]], x_key: str, y_key: str,
                   n_grid: int = 200):
    """Interpolate each seed's (x, y) onto a common grid.

    Returns (x_grid, stacked_y) where stacked_y has shape (n_seeds, n_grid).
    The grid spans the intersection of per-seed [x_min, x_max] to avoid
    extrapolation, and is uniformly spaced."""
    xs, ys = [], []
    for d in seed_data:
        if x_key not in d or y_key not in d:
            continue
        x = np.asarray(d[x_key], dtype=float)
        y = np.asarray(d[y_key], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue
        # Ensure x is strictly increasing for interp1d
        order = np.argsort(x)
        xs.append(x[order])
        ys.append(y[order])
    if not xs:
        return None, None

    x_lo = max(x.min() for x in xs)
    x_hi = min(x.max() for x in xs)
    if x_hi <= x_lo:
        return None, None
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    stacked = np.stack([np.interp(x_grid, xi, yi) for xi, yi in zip(xs, ys)], axis=0)
    return x_grid, stacked


def _plot_panel(ax, env: str, variants: List[str], x_key: str, y_key: str,
                x_label: str, y_label: str, title: str, smooth: int):
    for variant in variants:
        cfg = VARIANT_CFG.get(variant)
        if cfg is None:
            log.warning(f"Unknown variant '{variant}' — skipping.")
            continue

        csv_paths = _discover_seed_csvs(env, variant)
        if not csv_paths:
            log.warning(f"[{variant}] no CSVs found at results/{env}/crl_{variant}_s*.csv")
            continue

        seed_data = [_load_csv(p) for p in csv_paths]
        seed_data = [d for d in seed_data if d is not None]
        n_seeds = len(seed_data)
        if n_seeds == 0:
            continue

        # Sanity: drop seeds missing required columns
        seed_data = [d for d in seed_data if x_key in d and y_key in d]
        if not seed_data:
            log.warning(f"[{variant}] no seeds have required columns.")
            continue

        x_grid, stacked = _interp_curves(seed_data, x_key, y_key, n_grid=300)
        if x_grid is None:
            continue

        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        # Smooth both mean and the band edges (after aggregation)
        mean_s = _smooth(mean, smooth)
        lo_s = _smooth(mean - std, smooth)
        hi_s = _smooth(mean + std, smooth)

        # x-axis scaling for env_steps
        x_plot = x_grid / 1e6 if x_key == "env_steps" else x_grid

        # Sparse markers
        n_markers = 12
        marker_every = max(1, len(x_plot) // n_markers)

        label = f"{cfg['label']} (n={n_seeds})"
        ax.plot(x_plot, mean_s,
                label=label, color=cfg["color"], linestyle=cfg["ls"],
                marker=cfg["marker"], markevery=marker_every,
                markersize=5, linewidth=2.0)

        if n_seeds >= 2:
            ax.fill_between(x_plot, lo_s, hi_s, color=cfg["color"], alpha=0.18, linewidth=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ant_ball")
    p.add_argument("--metric", default="eval/episode_success_any")
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variants to plot (default: all 5).")
    p.add_argument("--out_prefix", default=None,
                   help="Output filename prefix. Default: '<env>_crl_improvements'.")
    # --seed is accepted for backward-compat with older job scripts.
    # The plotter now auto-discovers ALL seeds found on disk, so this flag is ignored.
    p.add_argument("--seed", type=int, default=None,
                   help="[ignored] kept for backward-compat; all seeds auto-aggregated.")
    args = p.parse_args()
    if args.seed is not None:
        log.info(f"--seed {args.seed} passed but ignored (plotter aggregates all seeds).")

    env = args.env
    variants = args.variants or list(VARIANT_CFG.keys())
    out_prefix = args.out_prefix or f"{env}_crl_improvements"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Report seed availability up front
    log.info("Seed discovery:")
    any_found = False
    for v in variants:
        paths = _discover_seed_csvs(env, v)
        seeds = [int(p.stem.split("_s")[-1]) for p in paths]
        label = VARIANT_CFG.get(v, {}).get("label", v)
        if seeds:
            any_found = True
            log.info(f"  {label:<25} seeds: {sorted(seeds)}")
        else:
            log.info(f"  {label:<25} (none found)")
    if not any_found:
        log.error("No CSVs found for any variant. Run training first.")
        return

    # 1×2 figure: env_steps + wall_time_min
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    env_title = ENV_DISPLAY.get(env, env)
    fig.suptitle(f"CRL Improvements on {env_title}", fontsize=13, y=1.02)

    _plot_panel(axes[0], env, variants,
                x_key="env_steps", y_key=args.metric,
                x_label="Environment steps (millions)",
                y_label=args.metric, title="vs. environment steps",
                smooth=args.smooth)
    _plot_panel(axes[1], env, variants,
                x_key="wall_time_min", y_key=args.metric,
                x_label="Wall-clock time (minutes)",
                y_label=args.metric, title="vs. wall-clock time",
                smooth=args.smooth)

    fig.tight_layout()
    png_path = FIGURES_DIR / f"{out_prefix}.png"
    pdf_path = FIGURES_DIR / f"{out_prefix}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved {png_path}")
    log.info(f"Saved {pdf_path}")

    # Summary table (final + best over all seeds)
    log.info("-" * 76)
    log.info(f"{'Variant':<25} {'#seeds':>7} {'final mean ± std':>22} {'best mean ± std':>22}")
    log.info("-" * 76)
    for v in variants:
        cfg = VARIANT_CFG.get(v)
        if cfg is None:
            continue
        paths = _discover_seed_csvs(env, v)
        seed_data = [_load_csv(p) for p in paths]
        seed_data = [d for d in seed_data if d is not None and args.metric in d]
        if not seed_data:
            continue
        finals = []
        bests = []
        for d in seed_data:
            y = np.asarray(d[args.metric], dtype=float)
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue
            finals.append(y[-1])
            bests.append(np.max(y))
        if not finals:
            continue
        finals = np.asarray(finals)
        bests = np.asarray(bests)
        log.info(
            f"{cfg['label']:<25} {len(finals):>7d} "
            f"{finals.mean():>10.3f} ± {finals.std():<8.3f} "
            f"{bests.mean():>10.3f} ± {bests.std():<8.3f}"
        )
    log.info("-" * 76)


if __name__ == "__main__":
    main()
