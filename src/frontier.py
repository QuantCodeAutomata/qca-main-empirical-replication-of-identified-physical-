"""Joint identified-set frontier for crash-risk pairs (q, l).

:class:`FrontierSolver` reconstructs the boundary of the identified set
``{(q, l)}`` using the support-function approach:

    h(u) = max_{(q,l) ∈ S} u_q·q + u_l·l   subject to admissibility.

Each call to :meth:`~FrontierSolver.solve_support_function` solves one SOCP
for a fixed direction ``u = (u_q, u_l)`` on the unit circle.  The inner
approximation is the convex hull of the optimal points ``(q_opt, l_opt)``
collected over all solved directions; the outer approximation is the
intersection of half-spaces ``{(q,l) : u·(q,l) ≤ h(u)}``.

:meth:`~FrontierSolver.build_frontier` adaptively refines the direction set,
starting from ``n_initial`` equally-spaced angles and bisecting the arc with
the largest outer–inner gap until the gap falls below ``gap_tol`` or
``max_directions`` is reached.
"""

from __future__ import annotations

import logging
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull

from src.socp_builder import SOCPBuilder

logger = logging.getLogger(__name__)

# Angle precision: two angles closer than this (radians) are considered equal.
_ANGLE_EPS: float = 1e-10


class FrontierSolver:
    """Reconstructs the joint identified-set frontier for (q, l).

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
    B:
        Upper bound on the normalised return grid.
    solver:
        CVXPY solver name (default ``'CLARABEL'``).
    """

    def __init__(
        self,
        cs: Any,
        quotes_df: Any,
        gamma: int = 2,
        K: float = 0.90,
        B: float = 5.0,
        solver: str = "CLARABEL",
    ) -> None:
        self.solver = solver
        self.builder = SOCPBuilder(cs, quotes_df, gamma=gamma, K=K, B=B)
        self._n = self.builder.n_cells

    # ------------------------------------------------------------------
    # Core SOCP: support function in a single direction
    # ------------------------------------------------------------------

    def solve_support_function(self, direction: np.ndarray) -> dict:
        """Maximise ``u_q·q + u_l·l`` subject to admissibility.

        Parameters
        ----------
        direction:
            Length-2 array ``[u_q, u_l]`` giving the direction on the unit
            circle.  Need not be unit-normalised; zero vectors are rejected.

        Returns
        -------
        dict
            ``{'support_value': float, 'q_opt': float, 'l_opt': float,
            'status': str}``
        """
        d = np.asarray(direction, dtype=float)
        if d.shape != (2,):
            raise ValueError(
                f"direction must be a length-2 array; got shape {d.shape}."
            )
        norm = np.linalg.norm(d)
        if norm < _ANGLE_EPS:
            raise ValueError("direction must be non-zero.")

        u_q, u_l = float(d[0]), float(d[1])

        mass = cp.Variable(self._n, nonneg=True)
        constraints = (
            self.builder.admissibility_constraints(mass)
            + self.builder.quote_constraints(mass)
        )

        q_expr = self.builder.crash_prob_expr(mass)
        l_expr = self.builder.crash_log_loss_expr(mass)

        prob = cp.Problem(cp.Maximize(u_q * q_expr + u_l * l_expr), constraints)

        try:
            prob.solve(solver=self.solver)
        except cp.SolverError as exc:
            logger.error("solve_support_function: SolverError: %s", exc)
            return {
                "support_value": np.nan,
                "q_opt": np.nan,
                "l_opt": np.nan,
                "status": f"solver_error: {exc}",
            }

        if prob.status not in ("optimal", "optimal_inaccurate") or mass.value is None:
            logger.debug(
                "solve_support_function: non-optimal status=%s for direction (%.3f, %.3f)",
                prob.status,
                u_q,
                u_l,
            )
            return {
                "support_value": np.nan,
                "q_opt": np.nan,
                "l_opt": np.nan,
                "status": prob.status if prob.status is not None else "unknown",
            }

        q_val = float(q_expr.value)
        l_val = float(l_expr.value)
        h_val = u_q * q_val + u_l * l_val

        return {
            "support_value": h_val,
            "q_opt": q_val,
            "l_opt": l_val,
            "status": prob.status,
        }

    # ------------------------------------------------------------------
    # Adaptive frontier construction
    # ------------------------------------------------------------------

    def build_frontier(
        self,
        n_initial: int = 32,
        max_directions: int = 128,
        gap_tol: float = 0.01,
    ) -> dict:
        """Build an inner/outer frontier approximation adaptively.

        Starts with ``n_initial`` equally-spaced unit-circle directions,
        solves the support function for each, then iteratively bisects the arc
        whose outer–inner gap is largest, until the global gap falls below
        ``gap_tol`` or the direction count reaches ``max_directions``.

        The **outer** support value in direction ``u`` is the SOCP optimal
        value ``h(u)``.  The **inner** support value is
        ``max_j u·(q_j, l_j)`` over all solved interior points.  The gap
        ``h(u) - inner_h(u)`` is non-negative and measures how far the inner
        polytope lies from the outer half-space boundary.

        Parameters
        ----------
        n_initial:
            Number of equally-spaced directions to solve initially.
        max_directions:
            Hard cap on total directions.
        gap_tol:
            Stop when the maximum outer–inner gap across all directions falls
            below this value.

        Returns
        -------
        dict
            ``{'inner_points': ndarray shape (m, 2), 'gap': float,
            'n_directions': int, 'support_values': dict[float, float]}``

            ``inner_points`` are the convex-hull vertices of all solved
            ``(q_opt, l_opt)`` pairs (or all such pairs if fewer than 3).
            ``support_values`` maps angle (radians) to outer support value.
        """
        angles = np.linspace(0.0, 2.0 * np.pi, n_initial, endpoint=False).tolist()

        # support_values[angle] = outer h(u)
        support_values: dict[float, float] = {}
        all_pts: list[tuple[float, float]] = []

        # Solve all initial directions
        for a in angles:
            res = self.solve_support_function(np.array([np.cos(a), np.sin(a)]))
            if res["status"] in ("optimal", "optimal_inaccurate"):
                support_values[a] = res["support_value"]
                all_pts.append((res["q_opt"], res["l_opt"]))

        if not all_pts:
            logger.error(
                "build_frontier: no feasible support-function evaluations."
            )
            return {
                "inner_points": np.empty((0, 2), dtype=float),
                "gap": np.nan,
                "n_directions": 0,
                "support_values": {},
            }

        pts = np.array(all_pts, dtype=float)

        # Adaptive refinement
        while len(support_values) < max_directions:
            sorted_angles = sorted(support_values.keys())
            n_a = len(sorted_angles)

            # Compute outer–inner gap for each direction
            gaps = {
                a: support_values[a] - float(np.max(pts @ np.array([np.cos(a), np.sin(a)])))
                for a in sorted_angles
            }
            max_gap = max(gaps.values())

            if max_gap <= gap_tol:
                break

            # Bisect the arc at the direction with the largest gap
            worst_angle = max(gaps, key=gaps.__getitem__)
            w_idx = sorted_angles.index(worst_angle)
            next_idx = (w_idx + 1) % n_a
            a1 = sorted_angles[w_idx]
            a2 = sorted_angles[next_idx]

            # Wrap-around-safe midpoint on the circle
            a2_wrap = a2 if a2 > a1 else a2 + 2.0 * np.pi
            new_angle = 0.5 * (a1 + a2_wrap) % (2.0 * np.pi)

            # Skip if numerically identical to an existing angle
            if any(abs(new_angle - existing) < _ANGLE_EPS for existing in support_values):
                logger.debug(
                    "build_frontier: bisected angle %.6f already exists; stopping.",
                    new_angle,
                )
                break

            res = self.solve_support_function(np.array([np.cos(new_angle), np.sin(new_angle)]))
            if res["status"] not in ("optimal", "optimal_inaccurate"):
                logger.debug(
                    "build_frontier: non-optimal status=%s at new angle=%.4f; stopping.",
                    res["status"],
                    new_angle,
                )
                break

            support_values[new_angle] = res["support_value"]
            all_pts.append((res["q_opt"], res["l_opt"]))
            pts = np.array(all_pts, dtype=float)

        # Final gap estimate
        final_gaps = [
            support_values[a] - float(np.max(pts @ np.array([np.cos(a), np.sin(a)])))
            for a in support_values
        ]
        final_gap = float(max(final_gaps)) if final_gaps else 0.0

        # Inner approximation: hull vertices when geometry allows it
        inner_pts = _convex_hull_vertices(pts)

        logger.debug(
            "build_frontier: n_directions=%d  gap=%.4e  inner_pts=%d",
            len(support_values),
            final_gap,
            inner_pts.shape[0],
        )

        return {
            "inner_points": inner_pts,
            "gap": final_gap,
            "n_directions": len(support_values),
            "support_values": support_values,
        }


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _convex_hull_vertices(pts: np.ndarray) -> np.ndarray:
    """Return the convex-hull vertices of *pts* (shape (m, 2)).

    Falls back to returning *pts* unchanged when fewer than 3 points exist or
    when the points are (nearly) collinear.
    """
    if pts.shape[0] < 3:
        return pts
    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        return pts
