"""Data-loading utilities for SPX option cross-sections.

Provides the :class:`CrossSection` container and functions that turn raw
OptionMetrics-style option data into normalised quote tables ready for
crash-risk estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CrossSection:
    """One (trade_date, maturity) observation normalised to call-price space.

    Attributes
    ----------
    trade_date:
        Observation date as ``'YYYY-MM-DD'``.
    maturity_date:
        Option expiry date as ``'YYYY-MM-DD'``.
    horizon:
        Target days-to-maturity used to select this cross-section.
    D:
        Discount factor ``exp(-r * T)``.
    F:
        Forward price ``S * exp(r * T)``.
    quotes:
        Normalised option quotes with columns
        ``['k', 'c_lo', 'c_hi', 'opt_type']``, where ``k = K / F`` and
        ``c = option_price / (D * F)``.  Put quotes are already converted to
        call-equivalent form via put-call parity.
    """

    trade_date: str
    maturity_date: str
    horizon: int
    D: float
    F: float
    quotes: pd.DataFrame = field(repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RAW_FILTERS = (
    "strike > 0 and bid >= 0 and ask > 0 and bid <= ask and volume > 0"
)


def _apply_raw_filters(df: pd.DataFrame) -> pd.DataFrame:
    return df.query(_RAW_FILTERS).copy()


def _keep_wednesdays(df: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col])
    return df[dates.dt.dayofweek == 2].copy()   # Wednesday == 2


def _normalise_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns k, c_lo, c_hi using vectorised put-call parity for puts."""
    D = df["discount_factor"]
    F = df["forward_price"]
    DF = D * F

    df = df.copy()
    df["k"] = df["strike"] / F

    # Base normalisation: price / (D * F)
    df["c_lo"] = df["bid"] / DF
    df["c_hi"] = df["ask"] / DF

    # Puts: add (1 - k) so they express the call-equivalent via parity
    is_put = df["option_type"] == "P"
    df.loc[is_put, "c_lo"] += 1.0 - df.loc[is_put, "k"]
    df.loc[is_put, "c_hi"] += 1.0 - df.loc[is_put, "k"]

    return df


def _filter_valid_normalised(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["c_lo"] >= 0) & (df["c_hi"] > 0) & (df["c_lo"] <= df["c_hi"])
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cross_sections(
    path: Optional[str] = None,
    horizons: list[int] = None,
) -> list[CrossSection]:
    """Load and normalise SPX option data into a list of :class:`CrossSection`.

    Parameters
    ----------
    path:
        Path to a Parquet file containing raw option data.  When *None* the
        synthetic dataset is loaded (or generated) via
        :func:`data.synthetic_generator.load_or_generate`.
    horizons:
        Target days-to-maturity to extract.  Defaults to ``[23, 30, 37]``.

    Returns
    -------
    list[CrossSection]
        One entry per (trade_date, horizon) pair for which a valid cross-
        section could be assembled.
    """
    if horizons is None:
        horizons = [23, 30, 37]

    # --- load ---
    if path is not None:
        raw_df = pd.read_parquet(path)
    else:
        from data.synthetic_generator import load_or_generate
        raw_df = load_or_generate()

    # --- quality filters ---
    raw_df = _apply_raw_filters(raw_df)
    raw_df = _keep_wednesdays(raw_df)

    if raw_df.empty:
        logger.warning("No valid Wednesday observations found in the dataset.")
        return []

    # --- normalise once for the whole dataset ---
    raw_df = _normalise_all(raw_df)
    raw_df = _filter_valid_normalised(raw_df)

    # Ensure integer days_to_maturity for arithmetic below
    raw_df["days_to_maturity"] = raw_df["days_to_maturity"].astype(int)

    cross_sections: list[CrossSection] = []

    for trade_date_str, day_group in raw_df.groupby("trade_date", sort=True):
        for horizon in horizons:
            # Find all rows whose maturity is within ±3 days of the target
            dtm_diff = (day_group["days_to_maturity"] - horizon).abs()
            close_mask = dtm_diff <= 3
            candidates = day_group[close_mask]

            if candidates.empty:
                logger.debug(
                    "No maturity within ±3 days of horizon %d for %s.",
                    horizon, trade_date_str,
                )
                continue

            # Select the single maturity closest to the target horizon
            best_dtm = candidates.loc[dtm_diff[close_mask].idxmin(), "days_to_maturity"]
            subset = candidates[candidates["days_to_maturity"] == best_dtm].copy()

            maturity_date_str = subset["maturity_date"].iloc[0]
            D = float(subset["discount_factor"].iloc[0])
            F = float(subset["forward_price"].iloc[0])

            quotes = (
                subset[["k", "c_lo", "c_hi", "option_type"]]
                .rename(columns={"option_type": "opt_type"})
                .reset_index(drop=True)
            )

            if quotes.empty:
                logger.warning(
                    "Empty quote table after normalisation for %s horizon %d.",
                    trade_date_str, horizon,
                )
                continue

            cross_sections.append(CrossSection(
                trade_date=trade_date_str,
                maturity_date=maturity_date_str,
                horizon=horizon,
                D=D,
                F=F,
                quotes=quotes,
            ))

    logger.info("Loaded %d cross-sections.", len(cross_sections))
    return cross_sections


def filter_otm(quotes: pd.DataFrame) -> pd.DataFrame:
    """Keep only out-of-the-money options.

    OTM puts satisfy ``k < 1``; OTM calls satisfy ``k > 1``.

    Parameters
    ----------
    quotes:
        DataFrame with at least columns ``['k', 'opt_type']``.

    Returns
    -------
    pd.DataFrame
        Filtered subset, index reset.
    """
    otm_puts = (quotes["opt_type"] == "P") & (quotes["k"] < 1.0)
    otm_calls = (quotes["opt_type"] == "C") & (quotes["k"] > 1.0)
    return quotes[otm_puts | otm_calls].reset_index(drop=True)
