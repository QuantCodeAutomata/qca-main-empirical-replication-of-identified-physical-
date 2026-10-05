"""Tests for src.socp_builder — SOCP variable builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import cvxpy as cp

from src.socp_builder import SOCPBuilder, SOCPBuilderDepth


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_quotes():
    """A small set of normalised call-equivalent quotes."""
    return pd.DataFrame({
        "k": [0.85, 0.88, 0.92, 0.95, 1.05],
        "c_lo": [0.018, 0.010, 0.008, 0.005, 0.003],
        "c_hi": [0.022, 0.013, 0.011, 0.008, 0.006],
        "opt_type": ["P", "P", "P", "P", "C"],
    })


@pytest.fixture
def builder_gamma2(simple_quotes):
    return SOCPBuilder(cs=None, quotes_df=simple_quotes, gamma=2, K=0.90, B=5.0)


@pytest.fixture
def builder_gamma1(simple_quotes):
    return SOCPBuilder(cs=None, quotes_df=simple_quotes, gamma=1, K=0.90, B=5.0)


# ─── Grid construction ────────────────────────────────────────────────────────

class TestGridConstruction:
    def test_fixed_breakpoints_included(self, builder_gamma2):
        """Grid must include 0.0, K=0.90, and B=5.0."""
        lo = builder_gamma2.grid_lo
        hi = builder_gamma2.grid_hi
        all_pts = np.concatenate([lo, [hi[-1]]])
        assert 0.0 in all_pts or all_pts[0] == pytest.approx(0.0, abs=1e-9)
        assert any(abs(x - 0.90) < 1e-9 for x in all_pts), "K=0.90 not in breakpoints"
        assert abs(hi[-1] - 5.0) < 1e-9, "B=5.0 should be the last breakpoint"

    def test_strike_breakpoints_included(self, builder_gamma2, simple_quotes):
        """All strikes in (0, B) should appear as breakpoints."""
        all_pts = np.concatenate([builder_gamma2.grid_lo, [builder_gamma2.grid_hi[-1]]])
        for k in simple_quotes["k"]:
            if 0 < k < 5.0:
                assert any(abs(x - k) < 1e-9 for x in all_pts), f"Strike {k} not in grid"

    def test_cells_are_contiguous(self, builder_gamma2):
        """Each cell's hi should equal the next cell's lo."""
        lo = builder_gamma2.grid_lo
        hi = builder_gamma2.grid_hi
        for i in range(len(lo) - 1):
            assert abs(hi[i] - lo[i + 1]) < 1e-9, f"Cell gap at i={i}"

    def test_n_cells_positive(self, builder_gamma2):
        assert builder_gamma2.n_cells > 0

    def test_invalid_gamma(self, simple_quotes):
        with pytest.raises(ValueError, match="gamma"):
            SOCPBuilder(cs=None, quotes_df=simple_quotes, gamma=3, K=0.90, B=5.0)

    def test_invalid_K_B(self, simple_quotes):
        with pytest.raises(ValueError):
            SOCPBuilder(cs=None, quotes_df=simple_quotes, gamma=2, K=5.0, B=5.0)


# ─── Constraint building ──────────────────────────────────────────────────────

class TestAdmissibilityConstraints:
    def test_constraints_are_list(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        constraints = builder_gamma2.admissibility_constraints(y0)
        assert isinstance(constraints, list)
        assert len(constraints) > 0

    def test_quote_constraints_after_admissibility(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        builder_gamma2.admissibility_constraints(y0)
        qc = builder_gamma2.quote_constraints(y0)
        assert isinstance(qc, list)

    def test_quote_constraints_before_admissibility_raises(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        with pytest.raises(RuntimeError):
            builder_gamma2.quote_constraints(y0)

    def test_crash_prob_expr_requires_admissibility_first(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        with pytest.raises(RuntimeError):
            builder_gamma2.crash_prob_expr(y0)

    def test_admissibility_sets_internal_variables(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        builder_gamma2.admissibility_constraints(y0)
        assert builder_gamma2._rho is not None
        assert builder_gamma2._y1 is not None
        assert builder_gamma2._y2 is not None
        assert builder_gamma2._aK is not None


# ─── Objective expressions ────────────────────────────────────────────────────

class TestObjectiveExpressions:
    def test_crash_prob_expr_is_expression(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        builder_gamma2.admissibility_constraints(y0)
        expr = builder_gamma2.crash_prob_expr(y0)
        assert isinstance(expr, cp.Expression)

    def test_crash_log_loss_expr_is_expression(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        builder_gamma2.admissibility_constraints(y0)
        expr = builder_gamma2.crash_log_loss_expr(y0)
        assert isinstance(expr, cp.Expression)

    def test_crash_shortfall_raises(self, builder_gamma2):
        y0 = cp.Variable(builder_gamma2.n_cells, nonneg=True)
        builder_gamma2.admissibility_constraints(y0)
        with pytest.raises(NotImplementedError):
            builder_gamma2.crash_shortfall_expr(y0)

    def test_gamma1_uses_y1_for_crash_prob(self, builder_gamma1):
        y0 = cp.Variable(builder_gamma1.n_cells, nonneg=True)
        builder_gamma1.admissibility_constraints(y0)
        expr = builder_gamma1.crash_prob_expr(y0)
        assert isinstance(expr, cp.Expression)

    def test_empty_quotes_builds_without_error(self):
        """Builder with empty quotes should still construct valid constraints."""
        empty = pd.DataFrame(columns=["k", "c_lo", "c_hi", "opt_type"])
        builder = SOCPBuilder(cs=None, quotes_df=empty, gamma=2, K=0.90, B=5.0)
        y0 = cp.Variable(builder.n_cells, nonneg=True)
        constraints = builder.admissibility_constraints(y0)
        assert len(constraints) > 0


# ─── Depth-normalised builder ─────────────────────────────────────────────────

class TestSOCPBuilderDepth:
    def test_builds_without_error(self, simple_quotes):
        bd = SOCPBuilderDepth(cs=None, quotes_df=simple_quotes, gamma=2, K=0.90, B=5.0)
        yh0 = cp.Variable(bd.n_cells, nonneg=True)
        constraints = bd.admissibility_constraints(yh0)
        assert len(constraints) > 0

    def test_depth_expr_returns_expression(self, simple_quotes):
        bd = SOCPBuilderDepth(cs=None, quotes_df=simple_quotes, gamma=2, K=0.90, B=5.0)
        yh0 = cp.Variable(bd.n_cells, nonneg=True)
        bd.admissibility_constraints(yh0)
        expr = bd.depth_expr(yh0)
        assert isinstance(expr, cp.Expression)

    def test_quote_constraints_after_admissibility(self, simple_quotes):
        bd = SOCPBuilderDepth(cs=None, quotes_df=simple_quotes, gamma=2, K=0.90, B=5.0)
        yh0 = cp.Variable(bd.n_cells, nonneg=True)
        bd.admissibility_constraints(yh0)
        qc = bd.quote_constraints(yh0)
        assert isinstance(qc, list)

    def test_grid_matches_base_builder(self, simple_quotes):
        bd = SOCPBuilderDepth(cs=None, quotes_df=simple_quotes, gamma=2, K=0.90, B=5.0)
        base = SOCPBuilder(cs=None, quotes_df=simple_quotes, gamma=2, K=0.90, B=5.0)
        assert bd.n_cells == base.n_cells
        np.testing.assert_allclose(bd.grid_lo, base.grid_lo)
        np.testing.assert_allclose(bd.grid_hi, base.grid_hi)
