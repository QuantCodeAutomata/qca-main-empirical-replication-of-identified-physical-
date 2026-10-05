"""Exp 2 – Joint identified-set frontier for (q, l).

Reconstructs the boundary of the joint identified set Θ_{q,l} using
support-function SOCP optimization for selected SPX cross sections.

Demonstrates that:
- The joint feasible set is a closed convex set, generally smaller than
  the Cartesian product of marginal intervals.
- Identification gains from additional quotes appear as inward frontier shifts.
- Frontier geometry differs across calm and stress dates.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_cross_sections
from src.frontier import FrontierSolver
from src.quote_sets import build_all_quote_sets

GAMMA = 2
K = 0.90
B = 5.0
SOLVER = "CLARABEL"
HORIZONS = [23, 30, 37]
# Select a subset of cross sections for frontier computation (expensive)
N_SELECTED = 5       # cross sections to plot per horizon
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment():
    print("=" * 70)
    print("Exp 2: Joint identified-set frontier for (q, l)")
    print("=" * 70)

    all_css = load_cross_sections(horizons=HORIZONS)
    print(f"Loaded {len(all_css)} cross sections")

    # Select representative cross sections: spread evenly through time
    selected = _select_representative(all_css)
    print(f"Selected {len(selected)} cross sections for frontier analysis")

    gap_records = []
    for cs in selected:
        print(f"\n  Processing {cs.trade_date} h={cs.horizon}d ...")
        qs = build_all_quote_sets(cs, K=K)
        quote_sets = ["pair", "local", "wing", "full_otm"]

        frontiers = {}
        for qs_name in quote_sets:
            quotes_df = qs[qs_name]
            if quotes_df.empty or len(quotes_df) < 2:
                continue
            try:
                solver = FrontierSolver(cs, quotes_df, gamma=GAMMA, K=K, B=B, solver=SOLVER)
                result = solver.build_frontier(n_initial=16, max_directions=64, gap_tol=0.005)
                frontiers[qs_name] = result
                gap_records.append({
                    "trade_date": cs.trade_date,
                    "horizon": cs.horizon,
                    "quote_set": qs_name,
                    "n_directions": result["n_directions"],
                    "gap": result["gap"],
                    "n_inner_points": len(result["inner_points"]),
                })
                print(
                    f"    {qs_name}: n_dir={result['n_directions']} "
                    f"gap={result['gap']:.4e} pts={len(result['inner_points'])}"
                )
            except Exception as exc:
                print(f"    {qs_name}: FAILED ({exc})")
                continue

        if frontiers:
            _plot_frontier(cs, frontiers, qs)

    # Save gap diagnostics
    gap_df = pd.DataFrame(gap_records)
    gap_df.to_csv(RESULTS_DIR / "exp2_gap_diagnostics.csv", index=False)

    # Table 5 analogue: summarise gap and direction counts
    _print_table5(gap_df)

    return gap_df


def _select_representative(all_css, n=N_SELECTED):
    """Select n cross sections per horizon, evenly spaced through time."""
    selected = []
    for h in HORIZONS:
        h_css = [cs for cs in all_css if cs.horizon == h]
        if not h_css:
            continue
        step = max(1, len(h_css) // n)
        selected.extend(h_css[::step][:n])
    return selected


def _plot_frontier(cs, frontiers: dict, qs: dict):
    """Plot nested frontiers for the given cross section."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"pair": "red", "local": "orange", "wing": "steelblue", "full_otm": "green"}
    labels = {"pair": "Pair", "local": "Local (8 puts)", "wing": "Wing (all puts)", "full_otm": "Full OTM"}

    for qs_name, result in frontiers.items():
        pts = result["inner_points"]
        if len(pts) < 2:
            continue
        # Scale to percentage points
        q_pts = pts[:, 0] * 100
        l_pts = pts[:, 1] * 100
        ax.fill(
            np.append(q_pts, q_pts[0]),
            np.append(l_pts, l_pts[0]),
            alpha=0.2, color=colors.get(qs_name, "grey")
        )
        ax.plot(
            np.append(q_pts, q_pts[0]),
            np.append(l_pts, l_pts[0]),
            color=colors.get(qs_name, "grey"),
            linewidth=1.5,
            label=labels.get(qs_name, qs_name),
        )

    ax.set_xlabel("Crash probability q (%)")
    ax.set_ylabel("Unconditional shortfall l (%)")
    ax.set_title(
        f"Joint identified set Θ_{{q,l}}\n"
        f"{cs.trade_date}, {cs.horizon}d horizon  (γ=2, K=0.90)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fname = f"exp2_frontier_{cs.trade_date}_{cs.horizon}d.png"
    fig.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Frontier plot saved: {fname}")


def _print_table5(gap_df: pd.DataFrame):
    """Table 5 analogue: adaptive-grid accuracy by quote set."""
    print("\n─── Table 5 analogue: Frontier approximation diagnostics ───")
    print(f"{'QSet':>12} {'N_dir median':>14} {'Gap median':>12} {'Gap 95th%':>12}")
    print("-" * 55)
    for qs_name in ["pair", "local", "wing", "full_otm"]:
        sub = gap_df[gap_df["quote_set"] == qs_name]["gap"].dropna()
        n_dir = gap_df[gap_df["quote_set"] == qs_name]["n_directions"].dropna()
        if sub.empty:
            continue
        print(
            f"{qs_name:>12}  {np.median(n_dir):>13.0f}  "
            f"{np.median(sub):>12.4e}  {np.percentile(sub, 95):>12.4e}"
        )
    gap_df.to_csv(RESULTS_DIR / "exp2_table5.csv", index=False)
    print(f"\nTable 5 saved to {RESULTS_DIR / 'exp2_table5.csv'}")


if __name__ == "__main__":
    run_experiment()
