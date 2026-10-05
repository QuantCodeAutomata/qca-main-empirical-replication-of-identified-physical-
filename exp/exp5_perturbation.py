"""Exp 5 – Quote-information attribution and perturbation robustness.

Studies which parts of the retained option cross-section drive identification
via:
1. Spread doubling (artificially widening bid-ask spreads)
2. Strike thinning (keeping every other strike)
3. Reverse-deletion: removing put-wing layers from the full OTM set

Expected:
- Spread doubling: material loosening (~0.79 pp on q endpoints)
- Strike thinning: small effect (~0.073 pp)
- Reverse-deletion nearby-six: essentially no widening (redundancy)
- Removing far put wing: moderate widening (~0.5 pp)
- Removing OTM calls: very small widening (~0.05-0.13 pp)
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
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_cross_sections
from src.quote_manipulations import (
    double_spreads,
    thin_strikes,
    remove_far_put_wing,
    remove_otm_calls,
    remove_nearby_six_puts,
)
from src.quote_sets import build_all_quote_sets, build_full_otm
from src.socp_q_l import compute_q_bounds
from src.socp_s import compute_s_bounds

GAMMA = 2
K = 0.90
B = 5.0
SOLVER = "CLARABEL"
HORIZONS = [23, 30, 37]
N_SPREAD_THIN = 8     # cross sections for spread/thinning experiments
N_DELETION = 20      # cross sections for reverse-deletion
N_JOBS = -1
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _solve_cs(cs, quotes_df, label: str) -> dict:
    """Run SOCP for one cross-section, return key metrics."""
    row = {"trade_date": cs.trade_date, "horizon": cs.horizon, "manipulation": label}
    if quotes_df is None or quotes_df.empty:
        return {**row, **dict.fromkeys(["q_lo","q_hi","s_lo","s_hi","width_q","width_s"], np.nan)}
    try:
        q = compute_q_bounds(cs, quotes_df, gamma=GAMMA, K=K, B=B, solver=SOLVER)
        s = compute_s_bounds(cs, quotes_df, gamma=GAMMA, K=K, B=B, solver=SOLVER)
        row.update({k: v for k, v in q.items() if not k.startswith("status")})
        row.update({k: v for k, v in s.items() if not k.startswith("status")})
    except Exception as exc:
        row["error"] = str(exc)
    return row


def _deterministic_subsample(all_css, n_per_horizon: int):
    selected = []
    for h in HORIZONS:
        h_css = sorted([cs for cs in all_css if cs.horizon == h], key=lambda c: c.trade_date)
        step = max(1, len(h_css) // n_per_horizon)
        selected.extend(h_css[::step][:n_per_horizon])
    return selected


def run_experiment():
    print("=" * 70)
    print("Exp 5: Quote-information attribution and perturbation robustness")
    print("=" * 70)

    all_css = load_cross_sections(horizons=HORIZONS)
    print(f"Loaded {len(all_css)} cross sections")

    # ─── Part 1 & 2: Spread doubling and strike thinning ─────────────────────
    subset_24 = _deterministic_subsample(all_css, N_SPREAD_THIN // len(HORIZONS))
    print(f"\nSubset for spread/thinning: {len(subset_24)} cross sections")

    spread_thin_tasks = []
    for cs in subset_24:
        qs = build_all_quote_sets(cs, K=K)
        full = qs["full_otm"]
        if full.empty:
            continue
        spread_thin_tasks.append((cs, full, "benchmark"))
        spread_thin_tasks.append((cs, double_spreads(full), "doubled_spreads"))
        spread_thin_tasks.append((cs, thin_strikes(full), "thinned_strikes"))

    results_st = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_solve_cs)(cs, qdf, lbl) for cs, qdf, lbl in spread_thin_tasks
    )
    df_st = pd.DataFrame(results_st)
    df_st.to_csv(RESULTS_DIR / "exp5_spread_thinning.csv", index=False)
    _print_perturbation_summary(df_st)

    # ─── Part 3: Reverse deletion ─────────────────────────────────────────────
    subset_180 = _deterministic_subsample(all_css, N_DELETION // len(HORIZONS))
    print(f"\nSubset for reverse deletion: {len(subset_180)} cross sections")

    deletion_tasks = []
    for cs in subset_180:
        qs = build_all_quote_sets(cs, K=K)
        full = qs["full_otm"]
        pair = qs["pair"]
        local = qs["local"]
        if full.empty:
            continue
        deletion_tasks.append((cs, full, "full_otm"))
        deletion_tasks.append((cs, remove_far_put_wing(full, local), "no_far_wing"))
        deletion_tasks.append((cs, remove_otm_calls(full), "no_calls"))
        deletion_tasks.append((cs, remove_nearby_six_puts(full, pair, local), "no_nearby_six"))

    results_del = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_solve_cs)(cs, qdf, lbl) for cs, qdf, lbl in deletion_tasks
    )
    df_del = pd.DataFrame(results_del)
    df_del.to_csv(RESULTS_DIR / "exp5_reverse_deletion.csv", index=False)
    _print_deletion_summary(df_del)
    _plot_deletion_widths(df_del)

    return df_st, df_del


def _print_perturbation_summary(df: pd.DataFrame):
    """Print spread-doubling and strike-thinning effect sizes."""
    print("\n─── Spread doubling and strike thinning: endpoint changes (median |Δ| %) ───")
    bench = df[df["manipulation"] == "benchmark"].set_index(["trade_date", "horizon"])

    for manip, label in [("doubled_spreads", "Doubled spreads"), ("thinned_strikes", "Thinned strikes")]:
        sub = df[df["manipulation"] == manip].set_index(["trade_date", "horizon"])
        delta_q_lo = (sub["q_lo"] - bench["q_lo"]).dropna().abs()
        delta_q_hi = (sub["q_hi"] - bench["q_hi"]).dropna().abs()
        delta_s_lo = (sub["s_lo"] - bench["s_lo"]).dropna().abs() if "s_lo" in sub else pd.Series(dtype=float)
        delta_s_hi = (sub["s_hi"] - bench["s_hi"]).dropna().abs() if "s_hi" in sub else pd.Series(dtype=float)

        print(f"\n  {label}:")
        print(f"    Median |Δq_lo| = {delta_q_lo.median()*100:.3f} pp")
        print(f"    Median |Δq_hi| = {delta_q_hi.median()*100:.3f} pp")
        if not delta_s_lo.empty:
            print(f"    Median |Δs_lo| = {delta_s_lo.median()*100:.3f} pp")
            print(f"    Median |Δs_hi| = {delta_s_hi.median()*100:.3f} pp")


def _print_deletion_summary(df: pd.DataFrame):
    """Print reverse-deletion width widening by horizon."""
    print("\n─── Reverse deletion: median width widening vs full_otm ───")
    print(f"{'Horizon':>8} {'Deletion':>20}  {'Δwidth_q (pp)':>15}")
    print("-" * 50)

    bench = df[df["manipulation"] == "full_otm"].set_index(["trade_date", "horizon"])
    rows = []
    for manip in ["no_far_wing", "no_calls", "no_nearby_six"]:
        sub = df[df["manipulation"] == manip].set_index(["trade_date", "horizon"])
        for h in HORIZONS:
            try:
                w_bench = bench.xs(h, level="horizon")["width_q"].dropna()
                w_sub = sub.xs(h, level="horizon")["width_q"].dropna()
                common = w_bench.index.intersection(w_sub.index)
                delta = (w_sub.loc[common] - w_bench.loc[common]).dropna()
                med = np.nanmedian(delta) * 100 if len(delta) > 0 else np.nan
                print(f"{h:>8d}d {manip:>20s}  {med:>13.3f}")
                rows.append({"horizon": h, "deletion": manip, "delta_width_q_pct": med})
            except Exception:
                pass

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp5_deletion_widening.csv", index=False)
    print(f"\nDeletion widening table saved to {RESULTS_DIR / 'exp5_deletion_widening.csv'}")


def _plot_deletion_widths(df: pd.DataFrame):
    """Bar chart of median q widths by deletion type and horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exp 5: Crash-probability interval width by deletion type", fontsize=12)
    manips = ["full_otm", "no_far_wing", "no_calls", "no_nearby_six"]
    labels = ["Full OTM", "No far wing", "No calls", "No nearby 6"]
    colors = ["steelblue", "orange", "green", "red"]

    for ax, h in zip(axes, HORIZONS):
        widths = []
        for m in manips:
            sub = df[(df["horizon"] == h) & (df["manipulation"] == m)]["width_q"].dropna()
            widths.append(np.nanmedian(sub) * 100 if len(sub) > 0 else 0)
        x = np.arange(len(manips))
        bars = ax.bar(x, widths, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylabel("Median width of q (%)")
        ax.set_title(f"{h}-day horizon")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_deletion_widths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDeletion widths figure saved to {RESULTS_DIR / 'exp5_deletion_widths.png'}")


if __name__ == "__main__":
    run_experiment()
