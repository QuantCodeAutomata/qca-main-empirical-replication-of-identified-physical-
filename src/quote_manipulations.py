"""Quote-set manipulations for sensitivity analysis.

Each function takes a quote DataFrame with columns
``['k', 'c_lo', 'c_hi', 'opt_type']`` and returns a modified copy, preserving
the column schema and returning rows sorted ascending by ``k``.

The manipulations are designed to study how the width of the crash-risk
identified set responds to changes in the information content of the
option quotes:

* :func:`double_spreads`          — widens every bid-ask spread (less information).
* :func:`thin_strikes`            — keeps every other strike (sparser grid).
* :func:`remove_far_put_wing`     — restricts to local puts plus OTM calls.
* :func:`remove_otm_calls`        — drops the entire right wing.
* :func:`remove_nearby_six_puts`  — removes the six local (non-pair) puts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected column order returned by every function.
_COLS = ["k", "c_lo", "c_hi", "opt_type"]


def _sort_reset(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* sorted by ``k`` ascending with a fresh integer index."""
    return df[_COLS].sort_values("k").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public manipulations
# ---------------------------------------------------------------------------

def double_spreads(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Double every bid-ask spread around the midpoint price.

    For each quote the spread ``s = c_hi - c_lo`` is doubled symmetrically:

        new_c_lo = mid - s   (clipped to 0 to preserve non-negativity)
        new_c_hi = mid + s

    where ``mid = (c_lo + c_hi) / 2``.

    Parameters
    ----------
    quotes_df:
        Quote table with columns ``['k', 'c_lo', 'c_hi', 'opt_type']``.

    Returns
    -------
    pd.DataFrame
        Modified copy with doubled spreads, sorted by ``k``.
    """
    if quotes_df.empty:
        logger.warning("double_spreads: received empty quotes_df.")
        return quotes_df[_COLS].copy().reset_index(drop=True)

    out = quotes_df[_COLS].copy()
    mid = (out["c_lo"] + out["c_hi"]) / 2.0
    spread = out["c_hi"] - out["c_lo"]
    out["c_lo"] = (mid - spread).clip(lower=0.0)
    out["c_hi"] = mid + spread

    logger.debug(
        "double_spreads: %d quotes; mean spread before=%.5f, after=%.5f",
        len(out),
        float(spread.mean()),
        float((out["c_hi"] - out["c_lo"]).mean()),
    )
    return _sort_reset(out)


def thin_strikes(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Keep every other strike, applied separately to puts and calls.

    After sorting puts (``opt_type == 'P'``) ascending by ``k`` and calls
    (``opt_type == 'C'``) ascending by ``k``, rows at even positions
    (0, 2, 4, …) are retained for each group.  The two thinned subsets are
    then concatenated and returned sorted by ``k``.

    Parameters
    ----------
    quotes_df:
        Quote table with columns ``['k', 'c_lo', 'c_hi', 'opt_type']``.

    Returns
    -------
    pd.DataFrame
        Thinned quote table sorted by ``k``.
    """
    if quotes_df.empty:
        logger.warning("thin_strikes: received empty quotes_df.")
        return quotes_df[_COLS].copy().reset_index(drop=True)

    puts = (
        quotes_df[quotes_df["opt_type"] == "P"]
        .sort_values("k")
        .reset_index(drop=True)
    )
    calls = (
        quotes_df[quotes_df["opt_type"] == "C"]
        .sort_values("k")
        .reset_index(drop=True)
    )

    puts_thin = puts.iloc[::2]
    calls_thin = calls.iloc[::2]

    out = pd.concat([puts_thin, calls_thin], ignore_index=True)

    logger.debug(
        "thin_strikes: %d puts -> %d, %d calls -> %d",
        len(puts), len(puts_thin),
        len(calls), len(calls_thin),
    )
    return _sort_reset(out)


def remove_far_put_wing(
    quotes_df: pd.DataFrame,
    local_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only the local puts plus all OTM calls; drop the far put wing.

    Retains two subsets of *quotes_df*:

    1. **Local puts** — put rows (``opt_type == 'P'``) whose normalised strike
       ``k`` appears in *local_df* (matched exactly on ``k``).
    2. **OTM calls**  — call rows (``opt_type == 'C'``) with ``k > 1``.

    All other put rows (the "far put wing") are discarded.

    Parameters
    ----------
    quotes_df:
        Full quote table (typically the *full_otm* set).
    local_df:
        Local quote set; its put ``k`` values define which puts are kept.

    Returns
    -------
    pd.DataFrame
        Filtered quote table sorted by ``k``.
    """
    if quotes_df.empty:
        logger.warning("remove_far_put_wing: received empty quotes_df.")
        return quotes_df[_COLS].copy().reset_index(drop=True)

    local_put_ks: set[float] = set(
        local_df.loc[local_df["opt_type"] == "P", "k"].tolist()
    )

    keep_puts = quotes_df[
        (quotes_df["opt_type"] == "P") & (quotes_df["k"].isin(local_put_ks))
    ]
    keep_calls = quotes_df[
        (quotes_df["opt_type"] == "C") & (quotes_df["k"] > 1.0)
    ]

    out = pd.concat([keep_puts, keep_calls], ignore_index=True)

    logger.debug(
        "remove_far_put_wing: %d total -> %d kept (%d local puts, %d otm calls)",
        len(quotes_df), len(out), len(keep_puts), len(keep_calls),
    )
    return _sort_reset(out)


def remove_otm_calls(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Remove all OTM calls (``k > 1``), retaining only OTM put rows.

    Parameters
    ----------
    quotes_df:
        Quote table with columns ``['k', 'c_lo', 'c_hi', 'opt_type']``.

    Returns
    -------
    pd.DataFrame
        Put-only quote table sorted by ``k``.
    """
    if quotes_df.empty:
        logger.warning("remove_otm_calls: received empty quotes_df.")
        return quotes_df[_COLS].copy().reset_index(drop=True)

    out = quotes_df[quotes_df["opt_type"] == "P"]

    logger.debug(
        "remove_otm_calls: %d total -> %d puts retained",
        len(quotes_df), len(out),
    )
    return _sort_reset(out)


def remove_nearby_six_puts(
    quotes_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    local_df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove the six local (non-pair) puts from the full-OTM quote set.

    The **six local puts** are the put rows in *local_df* whose normalised
    strike ``k`` does not appear in *pair_df* — i.e., the set difference
    ``local_puts_k − pair_puts_k``.  These strikes are then removed from
    *quotes_df*.

    Parameters
    ----------
    quotes_df:
        Full-OTM quote table to be filtered (all puts and calls).
    pair_df:
        Pair quote set (typically two puts bracketing the target strike *K*).
    local_df:
        Local quote set (pair union the six nearest puts); the non-pair
        elements identify the strikes to remove.

    Returns
    -------
    pd.DataFrame
        Filtered quote table with the six nearby puts removed, sorted by ``k``.
    """
    if quotes_df.empty:
        logger.warning("remove_nearby_six_puts: received empty quotes_df.")
        return quotes_df[_COLS].copy().reset_index(drop=True)

    pair_put_ks: set[float] = set(
        pair_df.loc[pair_df["opt_type"] == "P", "k"].tolist()
    )
    local_put_ks: set[float] = set(
        local_df.loc[local_df["opt_type"] == "P", "k"].tolist()
    )

    # The six local puts are those in local but not in pair
    nearby_six_ks = local_put_ks - pair_put_ks

    if len(nearby_six_ks) == 0:
        logger.warning(
            "remove_nearby_six_puts: local_df and pair_df share the same put "
            "strikes; no rows removed."
        )
        return _sort_reset(quotes_df)

    logger.debug(
        "remove_nearby_six_puts: removing %d nearby put strikes: %s",
        len(nearby_six_ks),
        sorted(nearby_six_ks),
    )

    # Drop matching put rows from quotes_df
    drop_mask = (quotes_df["opt_type"] == "P") & (quotes_df["k"].isin(nearby_six_ks))
    out = quotes_df[~drop_mask]

    logger.debug(
        "remove_nearby_six_puts: %d total -> %d retained",
        len(quotes_df), len(out),
    )
    return _sort_reset(out)
