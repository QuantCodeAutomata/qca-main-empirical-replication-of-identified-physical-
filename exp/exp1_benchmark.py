"""Exp 1 – Main empirical replication: identified physical crash-risk sets.

Computes identified intervals [q_lo, q_hi], [l_lo, l_hi], [s_lo, s_hi]
for SPX cross sections from 2013–2023 under the benchmark specification:
  gamma=2, K=0.90, B=5.0

Aggregates median endpoints and widths by horizon (23d, 30d, 37d) and
quote set (pair, local, wing, full_otm).  Reproduces Table 1 and the
time-series figure for crash-probability intervals.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure src is on the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_cross_sections
from src.bootstrap import bootstrap_width_reduction
from src.quote_sets import build_all_quote_sets
from src.regularity import verify_regularity
from src.socp_q_l import compute_q_bounds
from src.socp_s import compute_s_bounds

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
GAMMA = 2
K = 0.90
B = 5.0
SOLVER = "CLARABEL"
HORIZONS = [23, 30, 37]
QUOTE_SETS = ["pair", "local", "wing", "full_otm"]
N_JOBS = 4           # parallel workers
# Deterministic subsample: limit per-horizon to keep runtime tractable
N_PER_HORIZON = 50   # ≈ 150 cross sections × 4 quote sets × 6 solves ≈ 3 600 SOCPs
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _subsample_cross_sections(all_css, n_per_horizon: int):
    """Return at most n_per_horizon cross sections per horizon, evenly spaced."""
    selected = []
    for h in HORIZONS:
        h_css = sorted([cs for cs in all_css if cs.horizon == h],
                       key=lambda c: c.trade_date)
        step = max(1, len(h_css) // n_per_horizon)
        selected.extend(h_css[::step][:n_per_horizon])
    return selected


# ─── Per-cross-section worker ─────────────────────────────────────────────────

def _process_cs(cs, qs_name: str) -> dict:
    """Solve all four bounds for one cross section and quote set."""
    quotes = getattr(qs_name, None, None)
    # qs is passed as a pre-built dict
    return cs, qs_name


def _worker(args) -> dict:
    cs, qs_name, quotes_df = args
    row = {
        "trade_date": cs.trade_date,
        "horizon": cs.horizon,
        "quote_set": qs_name,
    }
    try:
        q_res = compute_q_bounds(cs, quotes_df, gamma=GAMMA, K=K, B=B, solver=SOLVER)
        s_res = compute_s_bounds(cs, quotes_df, gamma=GAMMA, K=K, B=B, solver=SOLVER)
    except Exception as exc:
        logger.warning("Worker failed: %s, %s, %s: %s", cs.trade_date, cs.horizon, qs_name, exc)
        return {**row, **dict.fromkeys(["q_lo","q_hi","l_lo","l_hi","s_lo","s_hi",
                                        "width_q","width_l","width_s"], np.nan)}

    row.update(q_res)
    row.update(s_res)
    return row


# ─── Main routine ─────────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("Exp 1: Benchmark identified crash-risk sets (gamma=2, K=0.90, B=5)")
    print("=" * 70)

    # Load cross sections for all horizons
    all_css = load_cross_sections(horizons=HORIZONS)
    print(f"Loaded {len(all_css)} cross sections across horizons {HORIZONS}")

    # Apply deterministic subsample to keep runtime tractable
    all_css = _subsample_cross_sections(all_css, N_PER_HORIZON)
    print(f"Using {len(all_css)} cross sections after subsampling "
          f"({N_PER_HORIZON} per horizon)")

    # Build tasks: (cs, qs_name, quotes_df)
    tasks = []
    for cs in all_css:
        qs = build_all_quote_sets(cs, K=K)
        for qs_name in QUOTE_SETS:
            quotes_df = qs[qs_name]
            if quotes_df.empty:
                continue
            tasks.append((cs, qs_name, quotes_df))

    print(f"Total SOCP tasks: {len(tasks)}")

    # Run in parallel
    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_worker)(t) for t in tasks
    )

    df = pd.DataFrame([r for r in results if r is not None])

    # Convert to percentage points for reporting
    for col in ["q_lo", "q_hi", "width_q", "l_lo", "l_hi", "width_l",
                "s_lo", "s_hi", "width_s"]:
        if col in df.columns:
            df[col + "_pct"] = df[col] * 100.0

    df.to_csv(RESULTS_DIR / "exp1_raw_results.csv", index=False)
    print(f"Raw results saved to {RESULTS_DIR / 'exp1_raw_results.csv'}")

    # ─── Table 1: Horizon × Quote-set medians ────────────────────────────────
    _print_table1(df)

    # ─── Bootstrap width-reduction CIs ───────────────────────────────────────
    _bootstrap_reduction(df)

    # ─── Figure 1: Weekly time series of q intervals ─────────────────────────
    _plot_q_timeseries(df)

    # ─── Figure 2: Width reduction by quote-set progression ──────────────────
    _plot_width_reduction(df)

    return df


def _print_table1(df: pd.DataFrame):
    """Print and save Table 1 (median endpoints and widths)."""
    print("\n─── Table 1: Median identified intervals by horizon and quote set ───")
    header = f"{'Horizon':>8} {'QSet':>10}  {'q_lo%':>8} {'q_hi%':>8} {'w_q%':>8}  {'l_lo%':>8} {'l_hi%':>8}  {'s_lo%':>8} {'s_hi%':>8} {'w_s%':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for h in HORIZONS:
        for qs in QUOTE_SETS:
            sub = df[(df["horizon"] == h) & (df["quote_set"] == qs)].copy()
            if sub.empty:
                continue
            row = {
                "horizon": h, "quote_set": qs,
                "q_lo_pct": np.nanmedian(sub["q_lo"]) * 100,
                "q_hi_pct": np.nanmedian(sub["q_hi"]) * 100,
                "w_q_pct":  np.nanmedian(sub["width_q"]) * 100,
                "l_lo_pct": np.nanmedian(sub["l_lo"]) * 100,
                "l_hi_pct": np.nanmedian(sub["l_hi"]) * 100,
                "s_lo_pct": np.nanmedian(sub["s_lo"]) * 100,
                "s_hi_pct": np.nanmedian(sub["s_hi"]) * 100,
                "w_s_pct":  np.nanmedian(sub["width_s"]) * 100,
                "n_obs":    sub["q_lo"].notna().sum(),
            }
            rows.append(row)
            print(
                f"{h:>8d} {qs:>10s}  "
                f"{row['q_lo_pct']:>8.3f} {row['q_hi_pct']:>8.3f} {row['w_q_pct']:>8.3f}  "
                f"{row['l_lo_pct']:>8.4f} {row['l_hi_pct']:>8.4f}  "
                f"{row['s_lo_pct']:>8.3f} {row['s_hi_pct']:>8.3f} {row['w_s_pct']:>8.3f}"
            )

    table_df = pd.DataFrame(rows)
    table_df.to_csv(RESULTS_DIR / "exp1_table1.csv", index=False)
    print(f"\nTable 1 saved to {RESULTS_DIR / 'exp1_table1.csv'}")


def _bootstrap_reduction(df: pd.DataFrame):
    """Bootstrap 95% CIs for median proportional width reduction."""
    print("\n─── Bootstrap 95% CIs for width reduction (pair→local→wing→full_otm) ───")
    bs_rows = []
    pairs = [("pair", "local"), ("local", "wing"), ("wing", "full_otm")]

    for h in HORIZONS:
        for coarser, finer in pairs:
            sub_c = df[(df["horizon"] == h) & (df["quote_set"] == coarser)][["trade_date","width_q"]].dropna()
            sub_f = df[(df["horizon"] == h) & (df["quote_set"] == finer)][["trade_date","width_q"]].dropna()

            merged = sub_c.merge(sub_f, on="trade_date", suffixes=("_c","_f"))
            if merged.empty:
                continue
            w_c = merged["width_q_c"].values
            w_f = merged["width_q_f"].values

            bs = bootstrap_width_reduction(w_c, w_f, n_bootstrap=2000, block_length=8, seed=42)
            print(
                f"  h={h:2d}d  {coarser}→{finer}: "
                f"median_reduction={bs['median_reduction']*100:.2f}%  "
                f"CI=[{bs['ci_lower']*100:.2f}%, {bs['ci_upper']*100:.2f}%]"
            )
            bs_rows.append({
                "horizon": h, "coarser": coarser, "finer": finer,
                "median_reduction_pct": bs["median_reduction"] * 100,
                "ci_lower_pct": bs["ci_lower"] * 100,
                "ci_upper_pct": bs["ci_upper"] * 100,
            })

    pd.DataFrame(bs_rows).to_csv(RESULTS_DIR / "exp1_bootstrap_reduction.csv", index=False)


def _plot_q_timeseries(df: pd.DataFrame):
    """Figure 1: Weekly q interval bands by horizon."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Exp 1: Weekly crash-probability intervals (full OTM, γ=2, K=0.90)", fontsize=13)

    for ax, h in zip(axes, HORIZONS):
        sub = (
            df[(df["horizon"] == h) & (df["quote_set"] == "full_otm")]
            .dropna(subset=["q_lo","q_hi"])
            .copy()
        )
        if sub.empty:
            ax.set_title(f"{h}-day (no data)")
            continue

        sub["trade_date"] = pd.to_datetime(sub["trade_date"])
        sub = sub.sort_values("trade_date")
        x = sub["trade_date"].values
        q_lo = sub["q_lo"].values * 100
        q_hi = sub["q_hi"].values * 100

        ax.fill_between(x, q_lo, q_hi, alpha=0.4, label="Identified interval")
        ax.plot(x, q_lo, color="steelblue", lw=0.8)
        ax.plot(x, q_hi, color="steelblue", lw=0.8)
        ax.set_ylabel("q (%)")
        ax.set_title(f"{h}-day horizon  (n={len(sub)})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_fig1_q_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure 1 saved to {RESULTS_DIR / 'exp1_fig1_q_timeseries.png'}")


def _plot_width_reduction(df: pd.DataFrame):
    """Figure 2: Width reduction by quote-set progression."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exp 1: Crash-probability width contraction by quote-set progression", fontsize=12)

    for ax, h in zip(axes, HORIZONS):
        widths = []
        for qs in QUOTE_SETS:
            sub = df[(df["horizon"] == h) & (df["quote_set"] == qs)]["width_q"].dropna()
            widths.append(np.nanmedian(sub) * 100 if len(sub) > 0 else np.nan)

        ax.plot(QUOTE_SETS, widths, marker="o", linewidth=2, markersize=8, color="steelblue")
        ax.set_title(f"{h}-day horizon")
        ax.set_xlabel("Quote set")
        ax.set_ylabel("Median width of q (pp)")
        ax.grid(alpha=0.3)
        ax.set_xticks(range(len(QUOTE_SETS)))
        ax.set_xticklabels(QUOTE_SETS, rotation=15, fontsize=9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_fig2_width_reduction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved to {RESULTS_DIR / 'exp1_fig2_width_reduction.png'}")


if __name__ == "__main__":
    run_experiment()
