"""Tests for src.normalize — option quote normalization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.normalize import normalize_quotes, parity_convert, filter_valid_quotes, split_otm


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_put_call_df():
    """Minimal raw option dataframe for testing."""
    return pd.DataFrame({
        "strike": [3800, 3900, 4000, 4100, 4200, 4300, 4400],
        "bid": [2.0, 5.0, 12.0, 20.0, 15.0, 8.0, 3.0],
        "ask": [4.0, 7.0, 14.0, 22.0, 17.0, 10.0, 5.0],
        "option_type": ["P", "P", "P", "P", "C", "C", "C"],
    })


# ─── normalize_quotes ─────────────────────────────────────────────────────────

class TestNormalizeQuotes:
    def test_output_columns(self, raw_put_call_df):
        result = normalize_quotes(raw_put_call_df, D=0.99, F=4050.0)
        assert set(["k", "c_lo", "c_hi", "opt_type"]).issubset(result.columns)

    def test_k_values(self, raw_put_call_df):
        F = 4050.0
        result = normalize_quotes(raw_put_call_df, D=0.99, F=F)
        expected_k = raw_put_call_df["strike"].values / F
        np.testing.assert_allclose(result["k"].values, expected_k, rtol=1e-10)

    def test_price_scaling(self, raw_put_call_df):
        D, F = 0.99, 4050.0
        DF = D * F
        result = normalize_quotes(raw_put_call_df, D=D, F=F)
        expected_c_lo = raw_put_call_df["bid"].values / DF
        expected_c_hi = raw_put_call_df["ask"].values / DF
        np.testing.assert_allclose(result["c_lo"].values, expected_c_lo, rtol=1e-10)
        np.testing.assert_allclose(result["c_hi"].values, expected_c_hi, rtol=1e-10)

    def test_output_length(self, raw_put_call_df):
        result = normalize_quotes(raw_put_call_df, D=0.99, F=4050.0)
        assert len(result) == len(raw_put_call_df)

    def test_option_type_preserved(self, raw_put_call_df):
        result = normalize_quotes(raw_put_call_df, D=0.99, F=4050.0)
        assert list(result["opt_type"]) == list(raw_put_call_df["option_type"])

    def test_zero_bid_allowed(self):
        """Zero bid should still produce valid c_lo=0 after normalization."""
        df = pd.DataFrame({
            "strike": [3800.0],
            "bid": [0.0],
            "ask": [2.0],
            "option_type": ["P"],
        })
        result = normalize_quotes(df, D=0.995, F=4000.0)
        assert result["c_lo"].iloc[0] == pytest.approx(0.0)


# ─── parity_convert ──────────────────────────────────────────────────────────

class TestParityConvert:
    def test_put_adjustment(self):
        """Put c_lo and c_hi should increase by (1 - k) after conversion."""
        df = pd.DataFrame({
            "k": [0.80, 0.90, 0.95],
            "c_lo": [0.01, 0.02, 0.03],
            "c_hi": [0.02, 0.03, 0.04],
            "opt_type": ["P", "P", "P"],
        })
        result = parity_convert(df)
        for i, k in enumerate([0.80, 0.90, 0.95]):
            adj = 1.0 - k
            assert result["c_lo"].iloc[i] == pytest.approx(df["c_lo"].iloc[i] + adj)
            assert result["c_hi"].iloc[i] == pytest.approx(df["c_hi"].iloc[i] + adj)

    def test_call_unchanged(self):
        """Call rows should not be modified by parity_convert."""
        df = pd.DataFrame({
            "k": [1.05, 1.10],
            "c_lo": [0.01, 0.005],
            "c_hi": [0.015, 0.008],
            "opt_type": ["C", "C"],
        })
        result = parity_convert(df)
        pd.testing.assert_frame_equal(result, df)

    def test_mixed_types(self):
        """Mixed P/C rows: only P rows should be adjusted."""
        df = pd.DataFrame({
            "k": [0.90, 1.10],
            "c_lo": [0.02, 0.01],
            "c_hi": [0.03, 0.02],
            "opt_type": ["P", "C"],
        })
        result = parity_convert(df)
        # Put row
        assert result["c_lo"].iloc[0] == pytest.approx(0.02 + (1.0 - 0.90))
        assert result["c_hi"].iloc[0] == pytest.approx(0.03 + (1.0 - 0.90))
        # Call row unchanged
        assert result["c_lo"].iloc[1] == pytest.approx(0.01)
        assert result["c_hi"].iloc[1] == pytest.approx(0.02)


# ─── filter_valid_quotes ──────────────────────────────────────────────────────

class TestFilterValidQuotes:
    def test_removes_negative_c_lo(self):
        df = pd.DataFrame({
            "k": [0.90, 0.80],
            "c_lo": [-0.01, 0.02],
            "c_hi": [0.01, 0.03],
            "opt_type": ["P", "P"],
        })
        result = filter_valid_quotes(df)
        assert len(result) == 1
        assert result["k"].iloc[0] == pytest.approx(0.80)

    def test_removes_zero_c_hi(self):
        df = pd.DataFrame({
            "k": [0.90, 0.80],
            "c_lo": [0.00, 0.01],
            "c_hi": [0.00, 0.02],
            "opt_type": ["P", "P"],
        })
        result = filter_valid_quotes(df)
        assert len(result) == 1

    def test_removes_crossed_quotes(self):
        """c_lo > c_hi should be removed."""
        df = pd.DataFrame({
            "k": [0.90, 0.80],
            "c_lo": [0.05, 0.01],
            "c_hi": [0.03, 0.02],
            "opt_type": ["P", "P"],
        })
        result = filter_valid_quotes(df)
        assert len(result) == 1

    def test_valid_quotes_retained(self):
        df = pd.DataFrame({
            "k": [0.90, 0.80, 1.10],
            "c_lo": [0.01, 0.02, 0.005],
            "c_hi": [0.02, 0.03, 0.010],
            "opt_type": ["P", "P", "C"],
        })
        result = filter_valid_quotes(df)
        assert len(result) == 3


# ─── split_otm ────────────────────────────────────────────────────────────────

class TestSplitOtm:
    def test_split_types(self):
        df = pd.DataFrame({
            "k": [0.85, 0.90, 1.00, 1.05, 1.10],
            "c_lo": [0.01] * 5,
            "c_hi": [0.02] * 5,
            "opt_type": ["P", "P", "P", "C", "C"],
        })
        puts, calls = split_otm(df)
        assert (puts["opt_type"] == "P").all()
        assert (puts["k"] < 1.0).all()
        assert (calls["opt_type"] == "C").all()
        assert (calls["k"] > 1.0).all()

    def test_atm_excluded(self):
        df = pd.DataFrame({
            "k": [1.00],
            "c_lo": [0.01],
            "c_hi": [0.02],
            "opt_type": ["P"],
        })
        puts, calls = split_otm(df)
        assert puts.empty
        assert calls.empty

    def test_empty_input(self):
        df = pd.DataFrame(columns=["k", "c_lo", "c_hi", "opt_type"])
        puts, calls = split_otm(df)
        assert puts.empty
        assert calls.empty

    def test_mathematical_property_put_k_lt_1(self):
        """All returned put strikes must satisfy k < 1."""
        df = pd.DataFrame({
            "k": np.linspace(0.7, 1.3, 20),
            "c_lo": [0.01] * 20,
            "c_hi": [0.02] * 20,
            "opt_type": ["P"] * 10 + ["C"] * 10,
        })
        puts, calls = split_otm(df)
        if not puts.empty:
            assert (puts["k"] < 1.0).all()
        if not calls.empty:
            assert (calls["k"] > 1.0).all()
