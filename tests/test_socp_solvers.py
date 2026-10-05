"""Tests for src.socp_q_l and src.socp_s — identified-set solvers."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from data.data_loader import CrossSection
from src.socp_q_l import compute_q_bounds, compute_all_bounds
from src.socp_s import compute_s_bounds

SOLVER = "CLARABEL"
K = 0.90
B = 5.0


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cs_stub():
    """Minimal CrossSection stub (metadata only)."""
    return CrossSection(
        trade_date="2020-01-08",
        maturity_date="2020-01-29",
        horizon=23,
        D=0.998,
        F=3300.0,
        quotes=pd.DataFrame(),
    )


@pytest.fixture
def informative_quotes():
    """Well-separated bid-ask quotes near K=0.90."""
    return pd.DataFrame({
        "k": [0.85, 0.88, 0.91, 0.94, 1.05],
        "c_lo": [0.020, 0.012, 0.008, 0.004, 0.003],
        "c_hi": [0.025, 0.016, 0.011, 0.007, 0.006],
        "opt_type": ["P", "P", "P", "P", "C"],
    })


@pytest.fixture
def pair_quotes():
    """Minimal pair: one put below K and one above K."""
    return pd.DataFrame({
        "k": [0.88, 0.92],
        "c_lo": [0.012, 0.008],
        "c_hi": [0.016, 0.011],
        "opt_type": ["P", "P"],
    })


# ─── compute_q_bounds ─────────────────────────────────────────────────────────

class TestComputeQBounds:
    def test_returns_dict(self, cs_stub, informative_quotes):
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        assert isinstance(res, dict)

    def test_required_keys(self, cs_stub, informative_quotes):
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        for key in ["q_lo", "q_hi", "l_lo", "l_hi", "width_q", "width_l"]:
            assert key in res, f"Missing key: {key}"

    def test_q_lo_leq_q_hi(self, cs_stub, informative_quotes):
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        if not (math.isnan(res["q_lo"]) or math.isnan(res["q_hi"])):
            assert res["q_lo"] <= res["q_hi"] + 1e-6, "q_lo must be <= q_hi"

    def test_q_in_unit_interval(self, cs_stub, informative_quotes):
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        for key in ["q_lo", "q_hi"]:
            v = res[key]
            if not math.isnan(v):
                assert 0.0 - 1e-6 <= v <= 1.0 + 1e-6, f"{key}={v} outside [0,1]"

    def test_l_non_negative(self, cs_stub, informative_quotes):
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        for key in ["l_lo", "l_hi"]:
            v = res[key]
            if not math.isnan(v):
                assert v >= -1e-6, f"{key}={v} is negative"

    def test_l_bounded_by_K_times_q(self, cs_stub, informative_quotes):
        """l should satisfy l <= K * q_hi approximately."""
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        if not any(math.isnan(res[k]) for k in ["l_hi", "q_hi"]):
            assert res["l_hi"] <= K * res["q_hi"] + 1e-5, \
                f"l_hi={res['l_hi']} exceeds K*q_hi={K*res['q_hi']}"

    def test_width_q_non_negative(self, cs_stub, informative_quotes):
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        if not math.isnan(res["width_q"]):
            assert res["width_q"] >= 0.0

    def test_empty_quotes_returns_nan(self, cs_stub):
        empty = pd.DataFrame(columns=["k", "c_lo", "c_hi", "opt_type"])
        res = compute_q_bounds(cs_stub, empty, gamma=2, K=K, B=B, solver=SOLVER)
        assert math.isnan(res["q_lo"])
        assert math.isnan(res["q_hi"])

    def test_gamma1_returns_valid_bounds(self, cs_stub, informative_quotes):
        res = compute_q_bounds(cs_stub, informative_quotes, gamma=1, K=K, B=B, solver=SOLVER)
        if not math.isnan(res["q_lo"]):
            assert 0.0 - 1e-6 <= res["q_lo"] <= 1.0 + 1e-6

    def test_pair_gives_wider_interval(self, cs_stub, pair_quotes, informative_quotes):
        """Pair (fewer quotes) should give wider interval than full OTM."""
        res_pair = compute_q_bounds(cs_stub, pair_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        res_full = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        if all(not math.isnan(res_pair[k]) for k in ["width_q"]) and \
           all(not math.isnan(res_full[k]) for k in ["width_q"]):
            assert res_pair["width_q"] >= res_full["width_q"] - 1e-5, \
                "Pair should give wider or equal interval than full OTM"

    def test_compute_all_bounds_is_alias(self, cs_stub, informative_quotes):
        res1 = compute_q_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        res2 = compute_all_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B)
        assert set(res1.keys()) == set(res2.keys())


# ─── compute_s_bounds ─────────────────────────────────────────────────────────

class TestComputeSBounds:
    def test_returns_dict(self, cs_stub, informative_quotes):
        res = compute_s_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        assert isinstance(res, dict)

    def test_required_keys(self, cs_stub, informative_quotes):
        res = compute_s_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        assert "s_lo" in res
        assert "s_hi" in res

    def test_s_lo_leq_s_hi(self, cs_stub, informative_quotes):
        res = compute_s_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        s_lo, s_hi = res.get("s_lo", np.nan), res.get("s_hi", np.nan)
        if not (math.isnan(s_lo) or math.isnan(s_hi)):
            assert s_lo <= s_hi + 1e-6

    def test_s_bounded_by_K(self, cs_stub, informative_quotes):
        """Conditional depth s should be in [0, K] since loss cannot exceed crash threshold."""
        res = compute_s_bounds(cs_stub, informative_quotes, gamma=2, K=K, B=B, solver=SOLVER)
        s_lo, s_hi = res.get("s_lo", np.nan), res.get("s_hi", np.nan)
        if not math.isnan(s_lo):
            assert s_lo >= -1e-5, f"s_lo={s_lo} is negative"
        if not math.isnan(s_hi):
            assert s_hi <= K + 1e-5, f"s_hi={s_hi} exceeds K={K}"

    def test_empty_quotes(self, cs_stub):
        empty = pd.DataFrame(columns=["k", "c_lo", "c_hi", "opt_type"])
        res = compute_s_bounds(cs_stub, empty, gamma=2, K=K, B=B, solver=SOLVER)
        # Should return nan or empty dict, not raise
        assert isinstance(res, dict)
