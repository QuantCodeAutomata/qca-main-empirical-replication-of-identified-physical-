"""Regularity verification for crash-risk identified sets.

Implements the slack-maximisation programme that verifies Assumption 3.3
(regularity) for a given cross-section and quote set.  A cross-section is
*regular* when there exists an admissible risk-neutral measure Q that is
strictly interior to the identified set, has positive mass on the left-tail
interval U_minus = (0.3, 0.5), and positive mass on the right-side interval
U_plus = (1.2, 1.8).

The check is cast as a SOCP: maximise epsilon subject to

* standard admissibility constraints (handled by SOCPBuilder),
* quote constraints tightened by epsilon on both sides,
* total mass on cells overlapping U_minus >= epsilon,
* total mass on cells overlapping U_plus  >= epsilon.

A positive optimal epsilon (above ``slack_tol``) certifies regularity.
"""

from __future__ import annotations

import logging
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from src.socp_builder import SOCPBuilder

logger = logging.getLogger(__name__)

# Intervals required to carry strictly positive mass under Assumption 3.3.
_U_MINUS_LO: float = 0.3
_U_MINUS_HI: float = 0.5
_U_PLUS_LO: float = 1.2
_U_PLUS_HI: float = 1.8


def verify_regularity(
    cs: Any,
    quotes_df: pd.DataFrame,
    gamma: int = 2,
    K: float = 0.90,
    B: float = 5.0,
    slack_tol: float = 1e-6,
    solver: str = "CLARABEL",
) -> dict:
    """Verify Assumption 3.3 (regularity) for a cross-section.

    Checks existence of an admissible Q that satisfies:

    1. Strictly interior quote satisfaction (positive slack on every bid/ask
       constraint).
    2. Positive mass in the open interval U_minus = (0.3, 0.5).
    3. Positive mass in the open interval U_plus  = (1.2, 1.8).

    All three conditions are encoded as a single slack-maximisation SOCP.

    Parameters
    ----------
    cs:
        Cross-section object passed through to :class:`~src.socp_builder.SOCPBuilder`.
    quotes_df:
        DataFrame with columns ``['k', 'c_lo', 'c_hi', 'opt_type']``
        (forward-normalised, call-equivalent prices).
    gamma:
        Moment parameter forwarded to :class:`~src.socp_builder.SOCPBuilder`.
    K:
        Crash-probability threshold (normalised strike, default 0.90).
    B:
        Upper bound on the normalised return grid.
    slack_tol:
        Minimum optimal epsilon to declare the cross-section regular.
    solver:
        CVXPY solver name.

    Returns
    -------
    dict
        ``{'regular': bool, 'slack': float, 'status': str}``
    """
    if quotes_df is None or quotes_df.empty:
        logger.warning("verify_regularity: empty quotes_df; returning non-regular.")
        return {"regular": False, "slack": 0.0, "status": "empty_quotes"}

    try:
        builder = SOCPBuilder(cs, quotes_df, gamma=gamma, K=K, B=B)
    except Exception as exc:
        logger.error("verify_regularity: SOCPBuilder construction failed: %s", exc)
        return {"regular": False, "slack": 0.0, "status": f"build_error: {exc}"}

    n = builder.n_cells
    mass = cp.Variable(n, nonneg=True)
    epsilon = cp.Variable(nonneg=True)

    constraints: list[cp.Constraint] = []
    constraints += builder.admissibility_constraints(mass)
    constraints += builder.quote_constraints(mass, slack=epsilon)

    # --- U_minus = (0.3, 0.5): cells whose support overlaps the interval -----
    idx_minus = np.where(
        (builder.grid_hi > _U_MINUS_LO) & (builder.grid_lo < _U_MINUS_HI)
    )[0]
    if idx_minus.size > 0:
        constraints.append(cp.sum(mass[idx_minus]) >= epsilon)
    else:
        logger.warning(
            "verify_regularity: no grid cells cover U_minus=[%.1f, %.1f]; "
            "regularity check for left tail is skipped.",
            _U_MINUS_LO,
            _U_MINUS_HI,
        )

    # --- U_plus = (1.2, 1.8): cells whose support overlaps the interval ------
    idx_plus = np.where(
        (builder.grid_hi > _U_PLUS_LO) & (builder.grid_lo < _U_PLUS_HI)
    )[0]
    if idx_plus.size > 0:
        constraints.append(cp.sum(mass[idx_plus]) >= epsilon)
    else:
        logger.warning(
            "verify_regularity: no grid cells cover U_plus=[%.1f, %.1f]; "
            "regularity check for right region is skipped.",
            _U_PLUS_LO,
            _U_PLUS_HI,
        )

    prob = cp.Problem(cp.Maximize(epsilon), constraints)

    try:
        prob.solve(solver=solver)
    except cp.SolverError as exc:
        logger.error("verify_regularity: solver error: %s", exc)
        return {"regular": False, "slack": 0.0, "status": f"solver_error: {exc}"}

    slack_val = float(epsilon.value) if epsilon.value is not None else 0.0
    is_regular = (
        prob.status in ("optimal", "optimal_inaccurate") and slack_val > slack_tol
    )

    logger.debug(
        "verify_regularity: status=%s  slack=%.3e  regular=%s",
        prob.status,
        slack_val,
        is_regular,
    )

    return {
        "regular": is_regular,
        "slack": slack_val,
        "status": prob.status if prob.status is not None else "unknown",
    }
