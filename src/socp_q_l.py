"""Identified-set bounds for crash probability q and unconditional shortfall l.

Each cross-section / quote-set combination produces four SOCP solves:
q_lo, q_hi, l_lo, l_hi.  Results are validated for logical consistency.

Public API
----------
compute_q_bounds(cs, quotes_df, gamma, K, B, solver) -> dict
compute_all_bounds(cs, quotes_df, gamma, K, B)       -> dict  (alias)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

from src.socp_builder import SOCPBuilder

logger = logging.getLogger(__name__)

_ACCEPT_STATUSES = frozenset({"optimal", "optimal_inaccurate"})
_TOL = 1e-6  # numerical tolerance for validation checks


def _solve_single(
    builder: SOCPBuilder,
    objective_key: str,
    solver: str,
) -> dict:
    """Solve one SOCP bound (q_min, q_max, l_min, or l_max).

    Returns
    -------
    dict
        ``{'value': float, 'status': str, 'q': float, 'l': float, 'rho': float}``
    """
    n = builder.n_cells
    y0 = cp.Variable(n, nonneg=True, name="y0")

    admissibility = builder.admissibility_constraints(y0)
    quotes = builder.quote_constraints(y0)
    all_constraints = admissibility + quotes

    q_expr = builder.crash_prob_expr(y0)
    l_expr = builder.crash_log_loss_expr(y0)

    if objective_key == "q_min":
        objective = cp.Minimize(q_expr)
        target_expr = q_expr
    elif objective_key == "q_max":
        objective = cp.Maximize(q_expr)
        target_expr = q_expr
    elif objective_key == "l_min":
        objective = cp.Minimize(l_expr)
        target_expr = l_expr
    elif objective_key == "l_max":
        objective = cp.Maximize(l_expr)
        target_expr = l_expr
    else:
        raise ValueError(f"Unknown objective_key: {objective_key}")

    prob = cp.Problem(objective, all_constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except cp.SolverError as exc:
        logger.debug("_solve_single [%s]: SolverError: %s", objective_key, exc)
        return {"value": np.nan, "status": f"solver_error: {exc}", "q": np.nan, "l": np.nan, "rho": np.nan}

    status = prob.status if prob.status is not None else "unknown"

    if status not in _ACCEPT_STATUSES or target_expr.value is None:
        logger.debug("_solve_single [%s]: status=%s", objective_key, status)
        return {"value": np.nan, "status": status, "q": np.nan, "l": np.nan, "rho": np.nan}

    val = float(target_expr.value)
    q_val = float(q_expr.value) if q_expr.value is not None else np.nan
    l_val = float(l_expr.value) if l_expr.value is not None else np.nan
    rho_val = float(builder._rho.value) if builder._rho is not None and builder._rho.value is not None else np.nan

    return {"value": val, "status": status, "q": q_val, "l": l_val, "rho": rho_val}


def compute_q_bounds(
    cs: Any,
    quotes_df: pd.DataFrame,
    gamma: int = 2,
    K: float = 0.90,
    B: float = 5.0,
    solver: str = "CLARABEL",
) -> dict:
    """Compute identified intervals [q_lo, q_hi] and [l_lo, l_hi].

    Parameters
    ----------
    cs:
        CrossSection object (passed to SOCPBuilder for metadata).
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
        Keys: q_lo, q_hi, l_lo, l_hi, status_q_lo, status_q_hi, status_l_lo,
        status_l_hi, width_q, width_l.
    """
    results: dict = {
        "q_lo": np.nan, "q_hi": np.nan,
        "l_lo": np.nan, "l_hi": np.nan,
        "status_q_lo": "not_run", "status_q_hi": "not_run",
        "status_l_lo": "not_run", "status_l_hi": "not_run",
        "width_q": np.nan, "width_l": np.nan,
    }

    if quotes_df is None or quotes_df.empty:
        logger.warning("compute_q_bounds: empty quotes_df; returning NaN.")
        results.update({"status_q_lo": "empty_quotes", "status_q_hi": "empty_quotes",
                        "status_l_lo": "empty_quotes", "status_l_hi": "empty_quotes"})
        return results

    try:
        builder = SOCPBuilder(cs, quotes_df, gamma=gamma, K=K, B=B)
    except Exception as exc:
        logger.error("compute_q_bounds: SOCPBuilder failed: %s", exc)
        return results

    for obj_key, q_key, l_key, stat_key in [
        ("q_min", "q_lo", None, "status_q_lo"),
        ("q_max", "q_hi", None, "status_q_hi"),
        ("l_min", None, "l_lo", "status_l_lo"),
        ("l_max", None, "l_hi", "status_l_hi"),
    ]:
        res = _solve_single(builder, obj_key, solver)
        results[stat_key] = res["status"]
        if not np.isnan(res["value"]):
            if q_key:
                results[q_key] = res["value"]
            if l_key:
                results[l_key] = res["value"]

    # --- Validation ---
    q_lo, q_hi = results["q_lo"], results["q_hi"]
    l_lo, l_hi = results["l_lo"], results["l_hi"]

    def _in_range(v: float, lo: float, hi: float) -> bool:
        return lo - _TOL <= v <= hi + _TOL

    if not np.isnan(q_lo) and not _in_range(q_lo, 0.0, 1.0):
        logger.warning("q_lo=%.4f outside [0,1]; clamping.", q_lo)
        results["q_lo"] = float(np.clip(q_lo, 0.0, 1.0))
    if not np.isnan(q_hi) and not _in_range(q_hi, 0.0, 1.0):
        logger.warning("q_hi=%.4f outside [0,1]; clamping.", q_hi)
        results["q_hi"] = float(np.clip(q_hi, 0.0, 1.0))
    if not np.isnan(l_lo) and l_lo < -_TOL:
        logger.warning("l_lo=%.4e < 0; clamping.", l_lo)
        results["l_lo"] = 0.0
    if not np.isnan(l_hi) and not np.isnan(q_hi):
        upper_l = K * q_hi + _TOL
        if l_hi > upper_l:
            logger.warning("l_hi=%.4e > K*q_hi=%.4e; clamping.", l_hi, upper_l)
            results["l_hi"] = float(upper_l)

    # --- Width ---
    if not np.isnan(q_lo) and not np.isnan(q_hi):
        results["width_q"] = max(0.0, float(q_hi - q_lo))
    if not np.isnan(l_lo) and not np.isnan(l_hi):
        results["width_l"] = max(0.0, float(l_hi - l_lo))

    return results


def compute_all_bounds(
    cs: Any,
    quotes_df: pd.DataFrame,
    gamma: int = 2,
    K: float = 0.90,
    B: float = 5.0,
) -> dict:
    """Alias for :func:`compute_q_bounds`."""
    return compute_q_bounds(cs, quotes_df, gamma=gamma, K=K, B=B)
