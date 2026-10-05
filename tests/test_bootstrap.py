"""Tests for src.bootstrap — circular block bootstrap utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.bootstrap import circular_block_bootstrap, bootstrap_width_reduction


# ─── circular_block_bootstrap ─────────────────────────────────────────────────

class TestCircularBlockBootstrap:
    def test_returns_dict(self):
        data = np.arange(40, dtype=float)
        result = circular_block_bootstrap(data, n_bootstrap=100, block_length=8, seed=42)
        assert isinstance(result, dict)
        assert "median" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "bootstrap_medians" in result

    def test_median_is_correct(self):
        data = np.arange(40, dtype=float)
        result = circular_block_bootstrap(data, n_bootstrap=100, block_length=8, seed=42)
        assert result["median"] == pytest.approx(np.median(data), abs=1e-9)

    def test_ci_ordered(self):
        data = np.arange(40, dtype=float)
        result = circular_block_bootstrap(data, n_bootstrap=500, block_length=8, seed=42)
        assert result["ci_lower"] <= result["median"]
        assert result["median"] <= result["ci_upper"]

    def test_bootstrap_medians_length(self):
        data = np.arange(40, dtype=float)
        n_boot = 200
        result = circular_block_bootstrap(data, n_bootstrap=n_boot, block_length=8, seed=42)
        assert len(result["bootstrap_medians"]) == n_boot

    def test_reproducible_with_same_seed(self):
        data = np.arange(40, dtype=float)
        r1 = circular_block_bootstrap(data, n_bootstrap=100, block_length=8, seed=0)
        r2 = circular_block_bootstrap(data, n_bootstrap=100, block_length=8, seed=0)
        np.testing.assert_array_equal(r1["bootstrap_medians"], r2["bootstrap_medians"])

    def test_different_seeds_differ(self):
        data = np.arange(40, dtype=float)
        r1 = circular_block_bootstrap(data, n_bootstrap=100, block_length=8, seed=1)
        r2 = circular_block_bootstrap(data, n_bootstrap=100, block_length=8, seed=2)
        assert not np.array_equal(r1["bootstrap_medians"], r2["bootstrap_medians"])

    def test_all_nan_input(self):
        data = np.full(20, np.nan)
        result = circular_block_bootstrap(data, n_bootstrap=50, block_length=4, seed=0)
        assert np.isnan(result["median"])
        assert np.isnan(result["ci_lower"])
        assert np.isnan(result["ci_upper"])

    def test_partial_nan_dropped(self):
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = circular_block_bootstrap(data, n_bootstrap=100, block_length=2, seed=0)
        # Median of non-NaN values = 3.0 (median of [1,2,4,5])
        assert not np.isnan(result["median"])

    def test_block_length_one(self):
        data = np.linspace(0, 1, 50)
        result = circular_block_bootstrap(data, n_bootstrap=100, block_length=1, seed=0)
        assert not np.isnan(result["median"])

    def test_single_observation(self):
        data = np.array([3.14])
        result = circular_block_bootstrap(data, n_bootstrap=50, block_length=1, seed=0)
        assert result["median"] == pytest.approx(3.14)
        assert result["ci_lower"] == pytest.approx(3.14)
        assert result["ci_upper"] == pytest.approx(3.14)


# ─── bootstrap_width_reduction ────────────────────────────────────────────────

class TestBootstrapWidthReduction:
    def test_returns_dict(self):
        wc = np.full(50, 5.0) + np.random.default_rng(0).normal(0, 0.2, 50)
        wf = np.full(50, 1.5) + np.random.default_rng(1).normal(0, 0.1, 50)
        result = bootstrap_width_reduction(wc, wf, n_bootstrap=100, block_length=8, seed=42)
        assert isinstance(result, dict)
        assert "median_reduction" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_ci_ordered(self):
        wc = np.full(50, 5.0)
        wf = np.full(50, 2.5)
        result = bootstrap_width_reduction(wc, wf, n_bootstrap=200, block_length=8, seed=42)
        assert result["ci_lower"] <= result["ci_upper"] + 1e-9

    def test_positive_reduction_when_finer_narrower(self):
        wc = np.full(50, 5.0)
        wf = np.full(50, 1.5)
        result = bootstrap_width_reduction(wc, wf, n_bootstrap=100, block_length=8, seed=42)
        # median reduction = 1 - 1.5/5.0 = 0.70
        assert result["median_reduction"] == pytest.approx(0.70, abs=1e-9)

    def test_equal_widths_zero_reduction(self):
        w = np.full(40, 3.0)
        result = bootstrap_width_reduction(w, w, n_bootstrap=100, block_length=8, seed=42)
        assert result["median_reduction"] == pytest.approx(0.0, abs=1e-9)

    def test_mismatched_shape_raises(self):
        wc = np.arange(30, dtype=float)
        wf = np.arange(20, dtype=float)
        with pytest.raises(ValueError, match="same shape"):
            bootstrap_width_reduction(wc, wf, n_bootstrap=50, block_length=4, seed=0)

    def test_nan_handling(self):
        wc = np.array([5.0, np.nan, 5.0, 5.0])
        wf = np.array([1.5, 1.5, np.nan, 1.5])
        result = bootstrap_width_reduction(wc, wf, n_bootstrap=50, block_length=2, seed=0)
        # Only one valid pair: (5.0, 1.5)
        assert result["median_reduction"] == pytest.approx(0.70, abs=1e-9)

    def test_all_nan_returns_nan(self):
        wc = np.full(10, np.nan)
        wf = np.full(10, np.nan)
        result = bootstrap_width_reduction(wc, wf, n_bootstrap=50, block_length=4, seed=0)
        assert np.isnan(result["median_reduction"])

    def test_ci_tighter_with_more_observations(self):
        """Wider sample should give tighter CI (probabilistically)."""
        rng = np.random.default_rng(99)
        wc_small = np.full(10, 5.0) + rng.normal(0, 0.5, 10)
        wf_small = np.full(10, 2.0) + rng.normal(0, 0.3, 10)
        wc_large = np.full(80, 5.0) + rng.normal(0, 0.5, 80)
        wf_large = np.full(80, 2.0) + rng.normal(0, 0.3, 80)
        r_small = bootstrap_width_reduction(wc_small, wf_small, n_bootstrap=500, block_length=4, seed=7)
        r_large = bootstrap_width_reduction(wc_large, wf_large, n_bootstrap=500, block_length=4, seed=7)
        width_small = r_small["ci_upper"] - r_small["ci_lower"]
        width_large = r_large["ci_upper"] - r_large["ci_lower"]
        assert width_large <= width_small * 1.5, \
            "Larger sample should produce (roughly) tighter CI"
