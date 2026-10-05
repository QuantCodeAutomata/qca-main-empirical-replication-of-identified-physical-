"""Exact SOCP builder for physical crash-risk identified sets.

Implements the Charnes-Cooper-scaled finite-dimensional SOCP representation
for the admissible risk-neutral measure set M(I) on [0, B] under a
power-utility pricing kernel with parameter gamma.

For gamma=2 (benchmark):
  - M(Q) = E_Q[X^2], Charnes-Cooper rho = 1/M(Q)
  - Scaled cell moments y0_j, y1_j, y2_j for all cells; y3_j for below-K cells
  - Global: sum y0 = rho, sum y1 = rho, sum y2 = 1
  - Crash probability: q = sum_{below} y2_j + K^2 * aK
  - Shortfall:         l = sum_{below} (K*y2_j - y3_j)

For gamma=1:
  - M(Q) = E_Q[X] = 1, rho = 1 (no scaling)
  - sum y0 = 1, sum y1 = 1
  - q = sum_{below} y1_j + K * aK
  - l = sum_{below} (K*y1_j - y2_j)

All truncated-moment conditions on cells are enforced via 2×2 PSD constraints
(Hausdorff moment problem on [a, b]).

Interface used by regularity.py, frontier.py, robustness.py:
  builder.n_cells            → int
  builder.grid_lo            → np.ndarray (n_cells,)
  builder.grid_hi            → np.ndarray (n_cells,)
  builder.admissibility_constraints(y0) → list[cp.Constraint]
  builder.quote_constraints(y0, slack)  → list[cp.Constraint]
  builder.crash_prob_expr(y0)           → cp.Expression (q)
  builder.crash_log_loss_expr(y0)       → cp.Expression (l)
  builder.crash_shortfall_expr(y0)      → NotImplementedError (use socp_s.py)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS_BREAK = 1e-10  # minimum cell width (narrower cells are merged)


class SOCPBuilder:
    """Builds the exact SOCP for crash-risk identification under gamma=2 (or gamma=1).

    Parameters
    ----------
    cs:
        CrossSection object (used for metadata only; not required to be non-None
        for pure solver tests).
    quotes_df:
        Forward-normalised call-equivalent quote table with columns
        ``['k', 'c_lo', 'c_hi', 'opt_type']``.
    gamma:
        Pricing-kernel curvature (1 or 2).
    K:
        Crash threshold (normalised strike, default 0.90).
    B:
        Upper support cap (default 5.0).
    """

    def __init__(
        self,
        cs: Any,
        quotes_df: pd.DataFrame,
        gamma: int = 2,
        K: float = 0.90,
        B: float = 5.0,
    ) -> None:
        if gamma not in (1, 2):
            raise ValueError(f"gamma must be 1 or 2; got {gamma}")
        if not (0 < K < B):
            raise ValueError(f"K must satisfy 0 < K < B; got K={K}, B={B}")

        self.gamma = gamma
        self.K = K
        self.B = B
        self._quotes_df = quotes_df.copy().reset_index(drop=True) if quotes_df is not None else pd.DataFrame()

        # Build breakpoints
        self._setup_grid()

        # Internal variables (set lazily by admissibility_constraints)
        self._rho: Optional[cp.Variable] = None
        self._y1: Optional[cp.Variable] = None
        self._y2: Optional[cp.Variable] = None
        self._y3: Optional[cp.Variable] = None
        self._aK: Optional[cp.Variable] = None

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _setup_grid(self) -> None:
        """Build cell breakpoints from {0, K, B} ∪ all strikes in (0, B)."""
        fixed = np.array([0.0, self.K, self.B])

        if not self._quotes_df.empty and "k" in self._quotes_df.columns:
            strikes = self._quotes_df["k"].values.astype(float)
            interior = strikes[(strikes > _EPS_BREAK) & (strikes < self.B - _EPS_BREAK)]
        else:
            interior = np.array([], dtype=float)

        pts = np.unique(np.concatenate([fixed, interior]))
        # Remove near-duplicate points
        keep = np.concatenate([[True], np.diff(pts) > _EPS_BREAK])
        pts = pts[keep]

        lo = pts[:-1]
        hi = pts[1:]

        self.n_cells: int = len(lo)
        self.grid_lo: np.ndarray = lo
        self.grid_hi: np.ndarray = hi

        # Classify cells
        # "below K" = hi <= K (entire cell is at or below K)
        # "above K" = lo >= K (entire cell is at or above K)
        self._below_K_mask: np.ndarray = hi <= self.K + _EPS_BREAK
        self._above_K_mask: np.ndarray = lo >= self.K - _EPS_BREAK
        self._below_K_idx: np.ndarray = np.where(self._below_K_mask)[0]
        self._above_K_idx: np.ndarray = np.where(self._above_K_mask)[0]

        # Threshold cell index: the cell whose lo ≈ K
        K_cell = np.where(np.abs(lo - self.K) < _EPS_BREAK)[0]
        self._K_cell_idx: Optional[int] = int(K_cell[0]) if len(K_cell) > 0 else None

        self._n_below: int = len(self._below_K_idx)
        logger.debug(
            "Grid: n_cells=%d, n_below=%d, K_cell=%s",
            self.n_cells,
            self._n_below,
            self._K_cell_idx,
        )

    # ------------------------------------------------------------------
    # Constraint builders
    # ------------------------------------------------------------------

    def admissibility_constraints(
        self, y0: cp.Variable
    ) -> List[cp.Constraint]:
        """Build all admissibility constraints for the SOCP.

        Creates fresh internal variables (y1, y2, y3, aK, rho) and returns
        a list of CVXPY constraints:

        * Global moment equalities (sum y0 = rho, sum y1 = rho, sum y2 = 1
          for gamma=2; or sum y0 = 1, sum y1 = 1 for gamma=1)
        * Hausdorff 2×2 PSD conditions on each cell
        * Threshold-atom residual conditions at K

        Parameters
        ----------
        y0:
            CVXPY variable of shape ``(n_cells,)`` representing scaled cell
            masses (ρ·Q(cell) for gamma=2, or Q(cell) for gamma=1).

        Returns
        -------
        list[cp.Constraint]
        """
        n = self.n_cells
        nb = self._n_below

        # --- Create fresh internal variables ---
        rho = cp.Variable(nonneg=True, name="rho")
        y1 = cp.Variable(n, nonneg=True, name="y1")
        y2 = cp.Variable(n, nonneg=True, name="y2")
        y3 = cp.Variable(nb, nonneg=True, name="y3")
        aK = cp.Variable(nonneg=True, name="aK")

        self._rho = rho
        self._y1 = y1
        self._y2 = y2
        self._y3 = y3
        self._aK = aK

        constraints: List[cp.Constraint] = []

        # --- Global moment equalities ---
        if self.gamma == 2:
            constraints.append(cp.sum(y0) == rho)
            constraints.append(cp.sum(y1) == rho)
            constraints.append(cp.sum(y2) == 1.0)
            constraints.append(rho >= 1.0 / self.B)
            constraints.append(rho <= 1.0)
        else:  # gamma=1: rho = 1 (pinned)
            constraints.append(rho == 1.0)
            constraints.append(cp.sum(y0) == 1.0)
            constraints.append(cp.sum(y1) == 1.0)
            # y2 is free (no sum constraint)

        # --- Hausdorff moment conditions per cell ---
        j_below = 0  # index into y3 for below-K cells
        for idx in range(n):
            a = float(self.grid_lo[idx])
            b = float(self.grid_hi[idx])
            width = b - a

            y0_j = y0[idx]
            y1_j = y1[idx]
            y2_j = y2[idx]

            is_below = bool(self._below_K_mask[idx])
            is_K_cell = (self._K_cell_idx is not None and idx == self._K_cell_idx)

            if is_below:
                y3_j = y3[j_below]
                j_below += 1
                # PSD1: [[y1-a*y0, y2-a*y1], [y2-a*y1, y3-a*y2]] >= 0
                constraints += _psd2(y1_j - a * y0_j, y2_j - a * y1_j, y3_j - a * y2_j)
                # PSD2: [[b*y0-y1, b*y1-y2], [b*y1-y2, b*y2-y3]] >= 0
                constraints += _psd2(b * y0_j - y1_j, b * y1_j - y2_j, b * y2_j - y3_j)
            else:
                if is_K_cell:
                    # Subtract threshold atom at K, apply conditions to residuals
                    y0_r = y0_j - aK
                    y1_r = y1_j - self.K * aK
                    y2_r = y2_j - self.K ** 2 * aK
                    # Non-negativity of residuals
                    constraints.append(y0_r >= 0)
                    # PSD on residuals: [[y0_r, y1_r],[y1_r, y2_r]] >= 0
                    constraints += _psd2_hankel(y0_r, y1_r, y2_r)
                    # Support: a*y0_r <= y1_r <= b*y0_r; y2_r <= (a+b)y1_r - ab*y0_r
                    if width > _EPS_BREAK:
                        constraints.append(y1_r >= a * y0_r)
                        constraints.append(y1_r <= b * y0_r)
                        constraints.append(y2_r <= (a + b) * y1_r - a * b * y0_r)
                else:
                    # Regular above-K cell
                    constraints += _psd2_hankel(y0_j, y1_j, y2_j)
                    if width > _EPS_BREAK:
                        constraints.append(y1_j >= a * y0_j)
                        constraints.append(y1_j <= b * y0_j)
                        constraints.append(y2_j <= (a + b) * y1_j - a * b * y0_j)

        return constraints

    def quote_constraints(
        self,
        y0: cp.Variable,
        slack: Union[float, cp.Variable] = 0.0,
    ) -> List[cp.Constraint]:
        """Build bid-ask quote constraints.

        For each retained quote i at normalised strike k_i:
            C_i(y) = Σ_{j: lo_j ≥ k_i} (y1_j − k_i · y0_j)
            rho · c_lo_i + slack ≤ C_i ≤ rho · c_hi_i − slack

        Parameters
        ----------
        y0:
            Cell-mass CVXPY variable (same object passed to
            :meth:`admissibility_constraints`).
        slack:
            Non-negative scalar or CVXPY variable for regularity checks.

        Returns
        -------
        list[cp.Constraint]
        """
        if self._y1 is None or self._rho is None:
            raise RuntimeError(
                "Call admissibility_constraints(y0) before quote_constraints(y0)."
            )

        y1 = self._y1
        rho = self._rho
        lo = self.grid_lo
        constraints: List[cp.Constraint] = []

        if self._quotes_df.empty:
            return constraints

        for _, row in self._quotes_df.iterrows():
            k_i = float(row["k"])
            c_lo_i = float(row["c_lo"])
            c_hi_i = float(row["c_hi"])

            # Cells fully above k_i: lo_j >= k_i (k_i is always a breakpoint)
            above_idx = np.where(lo >= k_i - _EPS_BREAK)[0]
            if len(above_idx) == 0:
                continue

            call_expr = cp.sum(y1[above_idx]) - k_i * cp.sum(y0[above_idx])

            lo_bound = rho * c_lo_i + slack
            hi_bound = rho * c_hi_i - slack
            constraints.append(call_expr >= lo_bound)
            constraints.append(call_expr <= hi_bound)

        return constraints

    # ------------------------------------------------------------------
    # Objective expressions
    # ------------------------------------------------------------------

    def crash_prob_expr(self, y0: cp.Variable) -> cp.Expression:
        """Physical crash probability q under the power-utility kernel.

        gamma=2: q = Σ_{j below K} y2_j + K² · aK
        gamma=1: q = Σ_{j below K} y1_j + K  · aK
        """
        if self._aK is None:
            raise RuntimeError("Call admissibility_constraints first.")

        if self.gamma == 2:
            return cp.sum(self._y2[self._below_K_idx]) + self.K ** 2 * self._aK
        else:  # gamma=1
            return cp.sum(self._y1[self._below_K_idx]) + self.K * self._aK

    def crash_log_loss_expr(self, y0: cp.Variable) -> cp.Expression:
        """Physical unconditional shortfall l under the power-utility kernel.

        gamma=2: l = Σ_{j below K} (K · y2_j − y3_j)
        gamma=1: l = Σ_{j below K} (K · y1_j − y2_j)
        """
        if self._y3 is None:
            raise RuntimeError("Call admissibility_constraints first.")

        if self.gamma == 2:
            below_idx = self._below_K_idx
            result = cp.sum(self.K * self._y2[below_idx]) - cp.sum(self._y3)
            return result
        else:  # gamma=1
            below_idx = self._below_K_idx
            return cp.sum(self.K * self._y1[below_idx]) - cp.sum(self._y2[below_idx])

    def crash_shortfall_expr(self, y0: cp.Variable) -> cp.Expression:
        """Conditional depth s = l/q — NOT directly representable as a convex expression.

        This raises NotImplementedError because s is a ratio and must be
        handled by the depth-normalised reformulation in :mod:`src.socp_s`.
        """
        raise NotImplementedError(
            "Conditional depth s = l/q requires the depth-normalised SOCP; "
            "use src.socp_s.compute_s_bounds instead."
        )


# ---------------------------------------------------------------------------
# Depth-Normalised Builder (for conditional depth s)
# ---------------------------------------------------------------------------

class SOCPBuilderDepth:
    """Depth-normalised SOCP for conditional depth s = K − A_3/A_2.

    Uses the reformulation:
        lambda = 1/A_2(Q),  dQ_hat = lambda · dQ
        Constraint: Σ_{j<K} ŷ2_j + K² · âK = 1  (normalises A_2 = 1)
        Objective:  s = Σ_{j<K} (K · ŷ2_j − ŷ3_j)   (linear in ŷ)

    Global constraints: Σ ŷ0 = lambda, Σ ŷ1 = lambda (E_{Q_hat}[X] = 1).
    Quote constraints: same structure as SOCPBuilder.
    """

    def __init__(
        self,
        cs: Any,
        quotes_df: pd.DataFrame,
        gamma: int = 2,
        K: float = 0.90,
        B: float = 5.0,
    ) -> None:
        # Reuse the main builder for grid construction
        self._base = SOCPBuilder(cs, quotes_df, gamma=gamma, K=K, B=B)
        self.gamma = gamma
        self.K = K
        self.B = B
        self.n_cells = self._base.n_cells
        self.grid_lo = self._base.grid_lo
        self.grid_hi = self._base.grid_hi
        self._quotes_df = self._base._quotes_df

        # Internal variables
        self._lam: Optional[cp.Variable] = None
        self._yh1: Optional[cp.Variable] = None
        self._yh2: Optional[cp.Variable] = None
        self._yh3: Optional[cp.Variable] = None
        self._aKh: Optional[cp.Variable] = None

    def admissibility_constraints(self, yh0: cp.Variable) -> List[cp.Constraint]:
        """Build depth-normalised admissibility constraints.

        Parameters
        ----------
        yh0:
            CVXPY variable of shape (n_cells,) for scaled cell masses.

        Returns
        -------
        list[cp.Constraint]
        """
        n = self.n_cells
        nb = self._base._n_below
        below_idx = self._base._below_K_idx
        K = self.K

        lam = cp.Variable(nonneg=True, name="lambda")
        yh1 = cp.Variable(n, nonneg=True, name="yh1")
        yh2 = cp.Variable(n, nonneg=True, name="yh2")
        yh3 = cp.Variable(nb, nonneg=True, name="yh3")
        aKh = cp.Variable(nonneg=True, name="aKh")

        self._lam = lam
        self._yh1 = yh1
        self._yh2 = yh2
        self._yh3 = yh3
        self._aKh = aKh

        constraints: List[cp.Constraint] = []

        # Global: sum yh0 = lambda, sum yh1 = lambda
        constraints.append(cp.sum(yh0) == lam)
        constraints.append(cp.sum(yh1) == lam)
        # Normalise: A_2 = 1 under the depth-normalised measure
        constraints.append(cp.sum(yh2[below_idx]) + K ** 2 * aKh == 1.0)
        constraints.append(lam >= 0)

        # Hausdorff conditions
        j_below = 0
        for idx in range(n):
            a = float(self.grid_lo[idx])
            b = float(self.grid_hi[idx])
            width = b - a
            is_below = bool(self._base._below_K_mask[idx])
            is_K_cell = (self._base._K_cell_idx is not None and idx == self._base._K_cell_idx)

            yh0_j = yh0[idx]
            yh1_j = yh1[idx]
            yh2_j = yh2[idx]

            if is_below:
                yh3_j = yh3[j_below]
                j_below += 1
                constraints += _psd2(yh1_j - a * yh0_j, yh2_j - a * yh1_j, yh3_j - a * yh2_j)
                constraints += _psd2(b * yh0_j - yh1_j, b * yh1_j - yh2_j, b * yh2_j - yh3_j)
            else:
                if is_K_cell:
                    yh0_r = yh0_j - aKh
                    yh1_r = yh1_j - K * aKh
                    yh2_r = yh2_j - K ** 2 * aKh
                    constraints.append(yh0_r >= 0)
                    constraints += _psd2_hankel(yh0_r, yh1_r, yh2_r)
                    if width > _EPS_BREAK:
                        constraints.append(yh1_r >= a * yh0_r)
                        constraints.append(yh1_r <= b * yh0_r)
                        constraints.append(yh2_r <= (a + b) * yh1_r - a * b * yh0_r)
                else:
                    constraints += _psd2_hankel(yh0_j, yh1_j, yh2_j)
                    if width > _EPS_BREAK:
                        constraints.append(yh1_j >= a * yh0_j)
                        constraints.append(yh1_j <= b * yh0_j)
                        constraints.append(yh2_j <= (a + b) * yh1_j - a * b * yh0_j)

        return constraints

    def quote_constraints(
        self,
        yh0: cp.Variable,
        slack: Union[float, cp.Variable] = 0.0,
    ) -> List[cp.Constraint]:
        """Quote constraints for depth-normalised measure."""
        if self._lam is None or self._yh1 is None:
            raise RuntimeError("Call admissibility_constraints first.")

        lam = self._lam
        yh1 = self._yh1
        lo = self.grid_lo
        constraints: List[cp.Constraint] = []

        for _, row in self._quotes_df.iterrows():
            k_i = float(row["k"])
            c_lo_i = float(row["c_lo"])
            c_hi_i = float(row["c_hi"])

            above_idx = np.where(lo >= k_i - _EPS_BREAK)[0]
            if len(above_idx) == 0:
                continue

            call_expr = cp.sum(yh1[above_idx]) - k_i * cp.sum(yh0[above_idx])
            lo_bound = lam * c_lo_i + slack
            hi_bound = lam * c_hi_i - slack
            constraints.append(call_expr >= lo_bound)
            constraints.append(call_expr <= hi_bound)

        return constraints

    def depth_expr(self, yh0: cp.Variable) -> cp.Expression:
        """s = Σ_{j<K} (K · ŷ2_j − ŷ3_j)  (linear under depth normalisation)."""
        if self._yh2 is None:
            raise RuntimeError("Call admissibility_constraints first.")
        below_idx = self._base._below_K_idx
        return cp.sum(self.K * self._yh2[below_idx]) - cp.sum(self._yh3)


# ---------------------------------------------------------------------------
# PSD constraint helpers
# ---------------------------------------------------------------------------

def _psd2(
    m00: Any, m01: Any, m11: Any
) -> List[cp.Constraint]:
    """Return [[ m00, m01 ], [ m01, m11 ]] >> 0 as a list of constraints.

    Uses the equivalent SOC formulation to avoid PSD overheads:
        m00 >= 0, m11 >= 0, m00*m11 >= m01^2  (via rotated SOC).
    """
    # Use direct PSD for 2x2 — CLARABEL handles this well
    M = cp.bmat([
        [cp.reshape(m00, (1, 1)), cp.reshape(m01, (1, 1))],
        [cp.reshape(m01, (1, 1)), cp.reshape(m11, (1, 1))],
    ])
    return [M >> 0]


def _psd2_hankel(y0: Any, y1: Any, y2: Any) -> List[cp.Constraint]:
    """Hankel [[y0, y1],[y1, y2]] >> 0."""
    M = cp.bmat([
        [cp.reshape(y0, (1, 1)), cp.reshape(y1, (1, 1))],
        [cp.reshape(y1, (1, 1)), cp.reshape(y2, (1, 1))],
    ])
    return [M >> 0]
