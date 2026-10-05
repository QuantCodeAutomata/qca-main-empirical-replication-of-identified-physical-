"""Option quote normalisation utilities.

Provides a composable set of transformations that convert raw bid/ask option
prices into the forward-normalised call-price space used by the crash-risk
estimators:

    c = option_price / (D * F)

For put quotes the additional put-call parity adjustment

    c_put_equiv = c_put + 1 - k,    k = K / F

expresses the put as its call equivalent, so that a single unified price
space can be used for both option types.

Typical pipeline
----------------
>>> quotes = normalize_quotes(raw_df, D=D, F=F)
>>> puts, calls = split_otm(quotes)
>>> puts_equiv = parity_convert(puts)          # convert OTM puts to call space
>>> valid_puts = filter_valid_quotes(puts_equiv)
>>> valid_calls = filter_valid_quotes(calls)
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Core transformations
# ---------------------------------------------------------------------------

def normalize_quotes(raw_df: pd.DataFrame, D: float, F: float) -> pd.DataFrame:
    """Normalise raw option bid/ask prices to forward-price units.

    Computes the dimensionless strike ratio ``k = K / F`` and divides each
    price by ``D * F`` to produce comparable ``c_lo``, ``c_hi`` values.
    Put quotes are **not** parity-adjusted here; call
    :func:`parity_convert` on the put subset afterwards.

    Parameters
    ----------
    raw_df:
        DataFrame containing at minimum the columns
        ``['strike', 'bid', 'ask', 'option_type']``.
    D:
        Discount factor ``exp(-r * T)`` for this cross-section.
    F:
        Forward price ``S * exp(r * T)`` for this cross-section.

    Returns
    -------
    pd.DataFrame
        Columns ``['k', 'c_lo', 'c_hi', 'opt_type']``.
    """
    DF = D * F
    out = pd.DataFrame({
        "k": raw_df["strike"].to_numpy() / F,
        "c_lo": raw_df["bid"].to_numpy() / DF,
        "c_hi": raw_df["ask"].to_numpy() / DF,
        "opt_type": raw_df["option_type"].to_numpy(),
    }, index=raw_df.index)
    return out.reset_index(drop=True)


def parity_convert(puts_df: pd.DataFrame) -> pd.DataFrame:
    """Convert put quotes to call-equivalent prices via put-call parity.

    Applies the parity identity

        c_call_equiv = p / (D * F) + 1 - k

    which, given that ``c_lo`` and ``c_hi`` already equal ``p / (D * F)``,
    reduces to adding ``1 - k`` to each bound:

        c_lo  +=  1 - k
        c_hi  +=  1 - k

    Parameters
    ----------
    puts_df:
        DataFrame with columns ``['k', 'c_lo', 'c_hi', 'opt_type']``
        whose rows are put quotes.  Rows with ``opt_type != 'P'`` are
        passed through unmodified.

    Returns
    -------
    pd.DataFrame
        Same schema as *puts_df*, with put rows adjusted for parity.
    """
    out = puts_df.copy()
    is_put = out["opt_type"] == "P"
    parity_adj = 1.0 - out.loc[is_put, "k"]
    out.loc[is_put, "c_lo"] += parity_adj
    out.loc[is_put, "c_hi"] += parity_adj
    return out


def filter_valid_quotes(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Remove quotes that violate basic no-arbitrage price bounds.

    A quote is retained only when all three conditions hold:

    * ``c_lo >= 0``   — lower bound is non-negative
    * ``c_hi > 0``    — upper bound is strictly positive
    * ``c_lo <= c_hi``— bid does not exceed ask

    Parameters
    ----------
    quotes_df:
        DataFrame with at minimum columns ``['c_lo', 'c_hi']``.

    Returns
    -------
    pd.DataFrame
        Filtered subset with the original column order, index reset.
    """
    mask = (
        (quotes_df["c_lo"] >= 0.0)
        & (quotes_df["c_hi"] > 0.0)
        & (quotes_df["c_lo"] <= quotes_df["c_hi"])
    )
    return quotes_df[mask].reset_index(drop=True)


def split_otm(
    quotes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a quote table into OTM puts and OTM calls.

    Out-of-the-money criterion:

    * Puts:  ``opt_type == 'P'`` and ``k < 1``
    * Calls: ``opt_type == 'C'`` and ``k > 1``

    At-the-money options (``k == 1``) are excluded from both subsets.

    Parameters
    ----------
    quotes_df:
        DataFrame with columns ``['k', 'opt_type']``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(otm_puts, otm_calls)``, each with index reset.
    """
    otm_puts = quotes_df[
        (quotes_df["opt_type"] == "P") & (quotes_df["k"] < 1.0)
    ].reset_index(drop=True)

    otm_calls = quotes_df[
        (quotes_df["opt_type"] == "C") & (quotes_df["k"] > 1.0)
    ].reset_index(drop=True)

    return otm_puts, otm_calls
