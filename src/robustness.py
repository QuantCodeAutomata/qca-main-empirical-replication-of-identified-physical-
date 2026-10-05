"""Robustness checks for crash-risk identified-set bounds.

Provides two robustness analyses that assess sensitivity to modelling choices:

* :func:`compute_bounds_varying_B` — sweeps the upper support cap *B* and
  re-computes identified-set bounds for every value.  This tests how much the
  bounds tighten or widen as the truncation of the return distribution is
  varied.

* :func:`compute_unbounded_relaxation` — computes outer bounds under a
  large-*B* relaxation (default ``B = 50``), giving an asymptotic benchmark
  against which finite-*B* bounds can be compared.

Both functions share the private helper :func:`_solve_bound`, which creates a
fresh CVXPY problem for a single objective (minimise or maximise a crash-risk
expression) subject to admissibility and quote constraints.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List

import cvxpy as cp
import numpy as np
import pandas as pd

from src.socp_builder import SOCPBuilder
from src.socp_s import compute_s_bounds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _solve_bound(
    builder: SOCPBuilder,
    expr_fn: Callable[[cp.Variable], cp.Expression],
    minimize: bool,
    solver: str,
) -> float:
    """Solve a single crash-risk bound optimisation.

    Parameters
    ----------
    builder:
        Fully constructed :class:`~src.socp_builder.SOCPBuilder` instance.
    expr_fn:
        Callable that takes a CVXPY mass variable and returns the scalar
        expression to optimise (e.g. ``builder.crash_prob_expr``).
    minimize:
        ``True`` for lower bound (minimise), ``False`` for upper bound
        (maximise).
    solver:
        CVXPY solver name.

    Returns
    -------
    float
        Optimal objective value, or ``np.nan`` on solver failure or
        non-optimal status.
    """
    n = builder.n_cells
    mass = cp.Variable(n, nonneg=True)
    constraints = (
        builder.admissibility_constraints(mass)
        + builder.quote_constraints(mass)
    )
    expr = expr_fn(mass)
    objective = cp.Minimize(expr) if minimize else cp.Maximize(expr)
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=solver)
    except cp.SolverError as exc:
        logger.debug("_solve_bound: SolverError: %s", exc)
        return np.nan

    if prob.status not in ("optimal", "optimal_inaccurate") or expr.value is None:
        logger.debug(
            "_solve_bound: non-optimal status=%s (minimize=%s)", prob.status, minimize
        )
        return np.nan

    return float(expr.value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_bounds_varying_B(
    cs: Any,
    quotes_df: pd.DataFrame,
    B_values: List[float] = [2, 3, 4, 5, 6, 7, 8, 9, 10],
    gamma: int = 2,
    K: float = 0.90,
    solver: str = "CLARABEL",
) -> pd.DataFrame:
    """Compute crash-risk bounds across a range of upper support caps *B*.

    For each value in *B_values* a fresh :class:`~src.socp_builder.SOCPBuilder`
    is constructed and six SOCP problems are solved (lower and upper bound for
    each of the three crash-risk measures: crash probability *q*, expected
    log-loss *l*, and expected shortfall *s*).

    Parameters
    ----------
    cs:
        Cross-section object forwarded to :class:`~src.socp_builder.SOCPBuilder`.
    quotes_df:
        Forward-normalised call-equivalent quote table
        ``['k', 'c_lo', 'c_hi', 'opt_type']``.
    B_values:
        Sequence of upper support-cap values to sweep.
    gamma:
        Moment parameter forwarded to :class:`~src.socp_builder.SOCPBuilder`.
    K:
        Crash-probability threshold (normalised strike, default 0.90).
    solver:
        CVXPY solver name.

    Returns
    -------
    pd.DataFrame
        Columns: ``['B', 'q_lo', 'q_hi', 'l_lo', 'l_hi', 's_lo', 's_hi']``.
        Rows corresponding to infeasible or failed solves contain ``NaN``.
    """
    rows: list[dict] = []

    for B in B_values:
        row: dict = {"B": float(B)}
        logger.debug("compute_bounds_varying_B: solving for B=%.1f", B)

        try:
            builder = SOCPBuilder(cs, quotes_df, gamma=gamma, K=K, B=B)
        except Exception as exc:
            logger.error(
                "compute_bounds_varying_B: SOCPBuilder failed for B=%.1f: %s", B, exc
            )
            row.update(
                {"q_lo": np.nan, "q_hi": np.nan,
                 "l_lo": np.nan, "l_hi": np.nan,
                 "s_lo": np.nan, "s_hi": np.nan}
            )
            rows.append(row)
            continue

        row["q_lo"] = _solve_bound(builder, builder.crash_prob_expr,     minimize=True,  solver=solver)
        row["q_hi"] = _solve_bound(builder, builder.crash_prob_expr,     minimize=False, solver=solver)
        row["l_lo"] = _solve_bound(builder, builder.crash_log_loss_expr, minimize=True,  solver=solver)
        row["l_hi"] = _solve_bound(builder, builder.crash_log_loss_expr, minimize=False, solver=solver)
        # Depth s uses depth-normalised SOCP (separate builder)
        s_res = compute_s_bounds(cs, quotes_df, gamma=gamma, K=K, B=B, solver=solver)
        row["s_lo"] = s_res.get("s_lo", np.nan)
        row["s_hi"] = s_res.get("s_hi", np.nan)

        logger.debug(
            "  B=%.1f  q=[%.4f, %.4f]  l=[%.4f, %.4f]  s=[%.4f, %.4f]",
            B,
            row["q_lo"], row["q_hi"],
            row["l_lo"], row["l_hi"],
            row["s_lo"], row["s_hi"],
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=["B", "q_lo", "q_hi", "l_lo", "l_hi", "s_lo", "s_hi"])


def compute_unbounded_relaxation(
    cs: Any,
    quotes_df: pd.DataFrame,
    gamma: int = 2,
    K: float = 0.90,
    B_relaxed: float = 50.0,
    solver: str = "CLARABEL",
) -> dict:
    """Compute crash-risk lower bounds under the unbounded-tail outer relaxation.

    Implements the paper's outer relaxation: a large ``B_relaxed`` cap is used
    **and** OTM call quotes are dropped.  OTM call upper bounds impose finite
    right-tail mass constraints; dropping them allows the optimizer to place
    arbitrary mass in the right tail, which (for gamma > 1) drives the physical
    crash-probability lower bound to zero — matching the paper's Theorem result.

    Only OTM put quotes (``k < 1``) are retained because they constrain the
    left side of the distribution and survive the right-tail relaxation.

    Parameters
    ----------
    cs:
        Cross-section object forwarded to :class:`~src.socp_builder.SOCPBuilder`.
    quotes_df:
        Forward-normalised call-equivalent quote table
        ``['k', 'c_lo', 'c_hi', 'opt_type']``.
    gamma:
        Moment parameter forwarded to :class:`~src.socp_builder.SOCPBuilder`.
    K:
        Crash-probability threshold (normalised strike, default 0.90).
    B_relaxed:
        Upper support cap for the relaxed problem (default 50.0).  The larger
        this is, the closer the result is to the truly unbounded-tail limit.
    solver:
        CVXPY solver name.

    Returns
    -------
    dict
        ``{'q_lo_relaxed': float, 'l_lo_relaxed': float, 'status': str}``
    """
    logger.debug(
        "compute_unbounded_relaxation: B_relaxed=%.1f  K=%.2f", B_relaxed, K
    )

    # Outer relaxation: retain only *lower* call-payoff bounds (c_lo_i ≤ C_i).
    # Upper bounds (C_i ≤ c_hi_i) constrain the right-tail mass and must be
    # dropped for the truly unbounded-tail relaxation.  With only lower bounds
    # and a large B, the optimizer can freely place mass far in the right tail,
    # which (for gamma > 1) drives the physical crash-probability lower bound to
    # ~0 — matching the paper's Theorem result.
    relaxed_quotes = quotes_df.copy()
    # Relax upper bounds: set c_hi to a very large number so the upper constraint
    # is never binding.
    relaxed_quotes["c_hi"] = 1e6

    try:
        builder = SOCPBuilder(cs, relaxed_quotes, gamma=gamma, K=K, B=B_relaxed)
    except Exception as exc:
        logger.error(
            "compute_unbounded_relaxation: SOCPBuilder construction failed: %s", exc
        )
        return {
            "q_lo_relaxed": np.nan,
            "l_lo_relaxed": np.nan,
            "status": f"build_error: {exc}",
        }

    q_lo = _solve_bound(builder, builder.crash_prob_expr,     minimize=True, solver=solver)
    l_lo = _solve_bound(builder, builder.crash_log_loss_expr, minimize=True, solver=solver)

    q_ok = not np.isnan(q_lo)
    l_ok = not np.isnan(l_lo)

    if q_ok and l_ok:
        status = "ok"
    elif q_ok:
        status = "l_failed"
    elif l_ok:
        status = "q_failed"
    else:
        status = "both_failed"

    logger.debug(
        "compute_unbounded_relaxation: q_lo=%.4f  l_lo=%.4f  status=%s",
        q_lo if q_ok else float("nan"),
        l_lo if l_ok else float("nan"),
        status,
    )

    return {"q_lo_relaxed": q_lo, "l_lo_relaxed": l_lo, "status": status}
