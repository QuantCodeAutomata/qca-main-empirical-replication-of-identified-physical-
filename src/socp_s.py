"""Identified-set bounds for conditional crash depth s = K − A_3/A_2.

Uses the depth-normalised reformulation:
    λ = 1/A_2(Q),  dQ_hat = λ · dQ
    Constraint: Σ_{j<K} ŷ2_j + K² · âK = 1   (normalises A_2 = 1)
    Objective:  s = Σ_{j<K} (K · ŷ2_j − ŷ3_j)  (linear in ŷ)

This makes s a linear objective in the scaled moments, enabling exact SOCP.

Public API
----------
compute_s_bounds(cs, quotes_df, gamma, K, B, solver) -> dict
"""

from __future__ import annotations

import logging
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from src.socp_builder import SOCPBuilderDepth

logger = logging.getLogger(__name__)

_ACCEPT_STATUSES = frozenset({"optimal", "optimal_inaccurate"})
_TOL = 1e-6


def _solve_s_single(
    builder: SOCPBuilderDepth,
    minimize: bool,
    solver: str,
) -> dict:
    """Solve one depth bound (s_min or s_max)."""
    n = builder.n_cells
    yh0 = cp.Variable(n, nonneg=True, name="yh0")

    admissibility = builder.admissibility_constraints(yh0)
    quotes = builder.quote_constraints(yh0)
    all_constraints = admissibility + quotes

    s_expr = builder.depth_expr(yh0)
    objective = cp.Minimize(s_expr) if minimize else cp.Maximize(s_expr)
    prob = cp.Problem(objective, all_constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except cp.SolverError as exc:
        logger.debug("_solve_s_single: SolverError: %s", exc)
        return {"value": np.nan, "status": f"solver_error: {exc}"}

    status = prob.status if prob.status is not None else "unknown"
    if status not in _ACCEPT_STATUSES or s_expr.value is None:
        logger.debug("_solve_s_single: status=%s", status)
        return {"value": np.nan, "status": status}

    return {"value": float(s_expr.value), "status": status}


def compute_s_bounds(
    cs: Any,
    quotes_df: pd.DataFrame,
    gamma: int = 2,
    K: float = 0.90,
    B: float = 5.0,
    solver: str = "CLARABEL",
) -> dict:
    """Compute identified interval [s_lo, s_hi] for conditional crash depth.

    Parameters
    ----------
    cs:
        CrossSection object.
    quotes_df:
        Forward-normalised call-equivalent quote table
        ``['k', 'c_lo', 'c_hi', 'opt_type']``.
    gamma:
        Pricing-kernel curvature (1 or 2).
    K:
        Crash threshold.
    B:
        Upper support cap.
    solver:
        CVXPY solver name.

    Returns
    -------
    dict
        Keys: s_lo, s_hi, status_s_lo, status_s_hi, width_s.
    """
    results: dict = {
        "s_lo": np.nan, "s_hi": np.nan,
        "status_s_lo": "not_run", "status_s_hi": "not_run",
        "width_s": np.nan,
    }

    if quotes_df is None or quotes_df.empty:
        logger.warning("compute_s_bounds: empty quotes_df.")
        results.update({"status_s_lo": "empty_quotes", "status_s_hi": "empty_quotes"})
        return results

    try:
        builder = SOCPBuilderDepth(cs, quotes_df, gamma=gamma, K=K, B=B)
    except Exception as exc:
        logger.error("compute_s_bounds: SOCPBuilderDepth failed: %s", exc)
        return results

    res_lo = _solve_s_single(builder, minimize=True, solver=solver)
    results["status_s_lo"] = res_lo["status"]
    if not np.isnan(res_lo["value"]):
        results["s_lo"] = float(np.clip(res_lo["value"], 0.0, K + _TOL))

    res_hi = _solve_s_single(builder, minimize=False, solver=solver)
    results["status_s_hi"] = res_hi["status"]
    if not np.isnan(res_hi["value"]):
        results["s_hi"] = float(np.clip(res_hi["value"], 0.0, K + _TOL))

    if not np.isnan(results["s_lo"]) and not np.isnan(results["s_hi"]):
        results["width_s"] = max(0.0, float(results["s_hi"] - results["s_lo"]))

    return results
