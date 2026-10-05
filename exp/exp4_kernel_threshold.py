"""Exp 4 – Kernel (gamma=1 vs 2) and threshold (K=0.85) robustness.

Part A: Compare identified sets under gamma=1 vs gamma=2 on common samples.
Part B: Recompute with more extreme threshold K=0.85 (gamma=2 fixed).

Expected: gamma=2 produces modestly tighter identified sets than gamma=1.
  Lower threshold K=0.85 → smaller q, wider depth uncertainty.
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
from src.quote_sets import build_full_otm
from src.socp_q_l import compute_q_bounds
from src.socp_s import compute_s_bounds

GAMMAS = [1, 2]
K_BENCHMARK = 0.90
K_EXTREME = 0.85
B = 5.0
SOLVER = "CLARABEL"
HORIZONS = [23, 30, 37]
N_JOBS = -1
N_PER_HORIZON = 12   # subsample cap for tractable runtime
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _subsample(all_css, n: int):
    selected = []
    for h in HORIZONS:
        h_css = sorted([c for c in all_css if c.horizon == h], key=lambda c: c.trade_date)
        step = max(1, len(h_css) // n)
        selected.extend(h_css[::step][:n])
    return selected


def _worker(cs, gamma: int, K: float) -> dict:
    quotes_df = build_full_otm(cs)
    row = {"trade_date": cs.trade_date, "horizon": cs.horizon, "gamma": gamma, "K": K}
    if quotes_df.empty:
        return {**row, **dict.fromkeys(["q_lo","q_hi","l_lo","l_hi","width_q","width_l","width_s"], np.nan)}
    try:
        q_res = compute_q_bounds(cs, quotes_df, gamma=gamma, K=K, B=B, solver=SOLVER)
        s_res = compute_s_bounds(cs, quotes_df, gamma=gamma, K=K, B=B, solver=SOLVER)
        row.update(q_res)
        row.update(s_res)
    except Exception as exc:
        row["error"] = str(exc)
    return row


def run_experiment():
    print("=" * 70)
    print("Exp 4: Kernel (γ=1 vs 2) and threshold (K=0.85) robustness")
    print("=" * 70)

    all_css = load_cross_sections(horizons=HORIZONS)
    print(f"Loaded {len(all_css)} cross sections")
    all_css = _subsample(all_css, N_PER_HORIZON)
    print(f"Using {len(all_css)} cross sections after subsampling ({N_PER_HORIZON}/horizon)")

    # ─── Part A: gamma comparison ─────────────────────────────────────────────
    print("\n[Part A] Computing gamma=1 and gamma=2 bounds on common sample ...")
    tasks_a = [(cs, g, K_BENCHMARK) for cs in all_css for g in GAMMAS]
    results_a = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_worker)(cs, g, K) for cs, g, K in tasks_a
    )
    df_a = pd.DataFrame([r for r in results_a if r])
    df_a.to_csv(RESULTS_DIR / "exp4_gamma_comparison.csv", index=False)

    _print_gamma_comparison(df_a)
    _plot_gamma_comparison(df_a)

    # ─── Part B: K=0.85 threshold ─────────────────────────────────────────────
    print("\n[Part B] Computing bounds under K=0.85 (gamma=2) ...")
    tasks_b = [(cs, 2, K_EXTREME) for cs in all_css]
    results_b = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_worker)(cs, g, K) for cs, g, K in tasks_b
    )
    df_b = pd.DataFrame([r for r in results_b if r])
    df_b.to_csv(RESULTS_DIR / "exp4_threshold_K085.csv", index=False)

    _print_threshold_comparison(df_b, df_a)

    return df_a, df_b


def _print_gamma_comparison(df: pd.DataFrame):
    """Table 2 analogue: width comparison gamma=1 vs gamma=2."""
    print("\n─── Table 2: Median q/l/s widths — gamma=1 vs gamma=2 ───")
    print(f"{'h':>4} {'gamma':>7}  {'w_q (pp)':>10} {'w_l (pp)':>10} {'w_s (pp)':>10}")
    print("-" * 46)
    rows = []
    for h in HORIZONS:
        for g in GAMMAS:
            sub = df[(df["horizon"] == h) & (df["gamma"] == g)]
            # Use common sample: cross sections where both gammas solved
            both = df[(df["horizon"] == h)].groupby("trade_date").filter(
                lambda x: x["gamma"].nunique() == 2
            )
            sub_c = both[(both["gamma"] == g)]
            w_q = np.nanmedian(sub_c["width_q"]) * 100 if not sub_c.empty else np.nan
            w_l = np.nanmedian(sub_c["width_l"]) * 100 if not sub_c.empty else np.nan
            w_s = np.nanmedian(sub_c["width_s"]) * 100 if not sub_c.empty else np.nan
            n = sub_c["width_q"].notna().sum()
            rows.append({"horizon": h, "gamma": g, "w_q_pct": w_q, "w_l_pct": w_l, "w_s_pct": w_s, "n": n})
            print(f"{h:>4d} {g:>7d}  {w_q:>10.3f} {w_l:>10.4f} {w_s:>10.3f}  (n={n})")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp4_table2_gamma.csv", index=False)
    print(f"\nTable 2 saved to {RESULTS_DIR / 'exp4_table2_gamma.csv'}")


def _print_threshold_comparison(df_85: pd.DataFrame, df_90: pd.DataFrame):
    """Table 3 analogue: K=0.85 vs K=0.90 medians."""
    print("\n─── Table 3: Median intervals — K=0.85 vs K=0.90 (gamma=2) ───")
    df_90_g2 = df_90[df_90["gamma"] == 2]
    rows = []
    for h in HORIZONS:
        for K_val, df_k in [(0.85, df_85), (0.90, df_90_g2)]:
            sub = df_k[df_k["horizon"] == h]
            row = {
                "horizon": h, "K": K_val,
                "q_lo": np.nanmedian(sub["q_lo"]) * 100,
                "q_hi": np.nanmedian(sub["q_hi"]) * 100,
                "w_q": np.nanmedian(sub["width_q"]) * 100,
                "w_s": np.nanmedian(sub["width_s"]) * 100,
                "n": sub["width_q"].notna().sum(),
            }
            rows.append(row)
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(RESULTS_DIR / "exp4_table3_threshold.csv", index=False)
    print(f"\nTable 3 saved to {RESULTS_DIR / 'exp4_table3_threshold.csv'}")


def _plot_gamma_comparison(df: pd.DataFrame):
    """Figure comparing q interval widths for gamma=1 vs gamma=2."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exp 4: q-interval width — γ=1 vs γ=2 (full OTM, K=0.90)", fontsize=12)

    for ax, h in zip(axes, HORIZONS):
        for g, color, ls in [(1, "darkorange", "--"), (2, "steelblue", "-")]:
            sub = df[(df["horizon"] == h) & (df["gamma"] == g)].dropna(subset=["width_q"])
            sub = sub.sort_values("trade_date")
            if sub.empty:
                continue
            ax.plot(
                pd.to_datetime(sub["trade_date"]), sub["width_q"] * 100,
                alpha=0.6, lw=0.8, color=color, linestyle=ls,
                label=f"γ={g}"
            )

        ax.set_title(f"{h}-day horizon")
        ax.set_xlabel("Date")
        ax.set_ylabel("Width of q (%)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_gamma_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGamma comparison figure saved to {RESULTS_DIR / 'exp4_gamma_comparison.png'}")


if __name__ == "__main__":
    run_experiment()
