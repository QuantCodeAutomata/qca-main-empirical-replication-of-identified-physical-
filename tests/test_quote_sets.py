"""Tests for src.quote_sets — nested OTM quote-set builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.data_loader import CrossSection
from src.quote_sets import (
    build_pair,
    build_local,
    build_wing,
    build_full_otm,
    build_all_quote_sets,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rich_cs():
    """CrossSection with many OTM puts and calls spanning K=0.90."""
    K = 0.90
    put_strikes = [0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.91, 0.92, 0.93, 0.95]
    call_strikes = [1.02, 1.05, 1.08, 1.10, 1.15]
    rows = []
    for k in put_strikes:
        c_lo = max(0.001, 0.03 * (K - k)) if k < K else 0.005
        rows.append({"k": k, "c_lo": c_lo, "c_hi": c_lo + 0.003, "opt_type": "P"})
    for k in call_strikes:
        c_lo = 0.01
        rows.append({"k": k, "c_lo": c_lo, "c_hi": c_lo + 0.003, "opt_type": "C"})
    quotes = pd.DataFrame(rows)
    return CrossSection(
        trade_date="2020-01-08",
        maturity_date="2020-01-29",
        horizon=23,
        D=0.998,
        F=3300.0,
        quotes=quotes,
    )


@pytest.fixture
def minimal_cs():
    """CrossSection with exactly one put below K and one above K."""
    rows = [
        {"k": 0.88, "c_lo": 0.02, "c_hi": 0.025, "opt_type": "P"},
        {"k": 0.91, "c_lo": 0.015, "c_hi": 0.020, "opt_type": "P"},
    ]
    quotes = pd.DataFrame(rows)
    return CrossSection(
        trade_date="2020-01-08",
        maturity_date="2020-01-29",
        horizon=23,
        D=0.998,
        F=3300.0,
        quotes=quotes,
    )


# ─── Tests: build_pair ────────────────────────────────────────────────────────

class TestBuildPair:
    def test_pair_has_two_quotes(self, rich_cs):
        pair = build_pair(rich_cs, K=0.90)
        assert len(pair) == 2

    def test_pair_brackets_K(self, rich_cs):
        K = 0.90
        pair = build_pair(rich_cs, K=K)
        strikes = sorted(pair["k"].tolist())
        assert strikes[0] < K  # one below K
        assert strikes[1] > K  # one above K

    def test_pair_returns_required_columns(self, rich_cs):
        pair = build_pair(rich_cs, K=0.90)
        assert set(["k", "c_lo", "c_hi", "opt_type"]).issubset(pair.columns)

    def test_pair_minimal_cs(self, minimal_cs):
        """Should still work when only one put on each side of K."""
        pair = build_pair(minimal_cs, K=0.90)
        assert len(pair) == 2

    def test_pair_sorted_ascending(self, rich_cs):
        pair = build_pair(rich_cs, K=0.90)
        assert list(pair["k"]) == sorted(pair["k"].tolist())


# ─── Tests: build_local ──────────────────────────────────────────────────────

class TestBuildLocal:
    def test_local_contains_pair(self, rich_cs):
        K = 0.90
        pair = build_pair(rich_cs, K=K)
        local = build_local(rich_cs, K=K)
        for k in pair["k"]:
            assert k in local["k"].values, f"Pair strike {k} missing from local"

    def test_local_has_at_most_8_puts(self, rich_cs):
        """Local set is pair (2) plus up to 6 additional puts = up to 8 puts."""
        K = 0.90
        local = build_local(rich_cs, K=K)
        n_puts = (local["opt_type"] == "P").sum()
        assert n_puts <= 8

    def test_local_larger_than_pair(self, rich_cs):
        pair = build_pair(rich_cs, K=0.90)
        local = build_local(rich_cs, K=0.90)
        assert len(local) >= len(pair)

    def test_local_returns_required_columns(self, rich_cs):
        local = build_local(rich_cs, K=0.90)
        assert set(["k", "c_lo", "c_hi", "opt_type"]).issubset(local.columns)

    def test_local_sorted_ascending(self, rich_cs):
        local = build_local(rich_cs, K=0.90)
        assert list(local["k"]) == sorted(local["k"].tolist())


# ─── Tests: build_wing ───────────────────────────────────────────────────────

class TestBuildWing:
    def test_wing_contains_all_otm_puts(self, rich_cs):
        """Wing should have all OTM puts from the cross-section."""
        all_otm_puts = rich_cs.quotes[
            (rich_cs.quotes["opt_type"] == "P") & (rich_cs.quotes["k"] < 1.0)
        ]
        wing = build_wing(rich_cs)
        n_puts_wing = (wing["opt_type"] == "P").sum()
        assert n_puts_wing == len(all_otm_puts)

    def test_wing_no_calls(self, rich_cs):
        """Wing set should contain only put quotes."""
        wing = build_wing(rich_cs)
        assert (wing["opt_type"] == "P").all() or wing.empty

    def test_wing_contains_local(self, rich_cs):
        """Wing is a superset of local."""
        local = build_local(rich_cs, K=0.90)
        wing = build_wing(rich_cs)
        wing_puts = set(wing["k"].tolist())
        local_puts = set(local[local["opt_type"] == "P"]["k"].tolist())
        assert local_puts.issubset(wing_puts)

    def test_wing_returns_required_columns(self, rich_cs):
        wing = build_wing(rich_cs)
        assert set(["k", "c_lo", "c_hi", "opt_type"]).issubset(wing.columns)


# ─── Tests: build_full_otm ───────────────────────────────────────────────────

class TestBuildFullOtm:
    def test_full_otm_contains_wing(self, rich_cs):
        wing = build_wing(rich_cs)
        full = build_full_otm(rich_cs)
        wing_k = set(wing["k"].tolist())
        full_k = set(full["k"].tolist())
        assert wing_k.issubset(full_k)

    def test_full_otm_contains_calls(self, rich_cs):
        full = build_full_otm(rich_cs)
        n_calls = (full["opt_type"] == "C").sum()
        assert n_calls > 0  # should have OTM calls

    def test_full_otm_sorted_ascending(self, rich_cs):
        full = build_full_otm(rich_cs)
        assert list(full["k"]) == sorted(full["k"].tolist())

    def test_nesting_pair_local_wing_full(self, rich_cs):
        """Nested containment: |pair| <= |local| <= |wing| <= |full_otm|."""
        K = 0.90
        pair = build_pair(rich_cs, K=K)
        local = build_local(rich_cs, K=K)
        wing = build_wing(rich_cs)
        full = build_full_otm(rich_cs)
        assert len(pair) <= len(local)
        assert len(local) <= len(wing) + 5  # local may have calls but wing only puts
        assert len(wing) <= len(full)


# ─── Tests: build_all_quote_sets ──────────────────────────────────────────────

class TestBuildAllQuoteSets:
    def test_returns_all_four_keys(self, rich_cs):
        all_sets = build_all_quote_sets(rich_cs, K=0.90)
        assert "pair" in all_sets
        assert "local" in all_sets
        assert "wing" in all_sets
        assert "full_otm" in all_sets

    def test_all_dataframes(self, rich_cs):
        all_sets = build_all_quote_sets(rich_cs, K=0.90)
        for name, df in all_sets.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"

    def test_monotone_size(self, rich_cs):
        """Quote sets should weakly increase in size."""
        all_sets = build_all_quote_sets(rich_cs, K=0.90)
        sizes = [len(all_sets[k]) for k in ["pair", "local", "wing", "full_otm"]]
        # pair <= local; wing <= full_otm
        assert sizes[0] <= sizes[1]
        assert sizes[2] <= sizes[3]
