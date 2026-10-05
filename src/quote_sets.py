"""Quote-set constructors for crash-risk estimation.

Each builder selects a specific subset of OTM option quotes from a
:class:`~data.data_loader.CrossSection` and returns it as a ``pd.DataFrame``
with columns ``['k', 'c_lo', 'c_hi', 'opt_type']``, sorted by ``k``.

The four quote sets span a spectrum of information content:

* **pair**     – two quotes bracketing the target strike *K*
* **local**    – pair ∪ six nearest OTM puts to *K*
* **wing**     – all OTM puts
* **full_otm** – all OTM puts and OTM calls
"""

from __future__ import annotations

import logging

import pandas as pd

from data.data_loader import CrossSection

logger = logging.getLogger(__name__)

# Column ordering preserved by all builders
_COLS = ["k", "c_lo", "c_hi", "opt_type"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _otm_puts(cs: CrossSection) -> pd.DataFrame:
    """Return OTM put rows (``k < 1``) from *cs.quotes*, sorted by ``k``."""
    q = cs.quotes
    mask = (q["opt_type"] == "P") & (q["k"] < 1.0)
    return q[mask].sort_values("k").reset_index(drop=True)


def _otm_calls(cs: CrossSection) -> pd.DataFrame:
    """Return OTM call rows (``k > 1``) from *cs.quotes*, sorted by ``k``."""
    q = cs.quotes
    mask = (q["opt_type"] == "C") & (q["k"] > 1.0)
    return q[mask].sort_values("k").reset_index(drop=True)


def _empty_quotes() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)


def _dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by ``k``, keep first occurrence, sort ascending."""
    return df.drop_duplicates(subset="k").sort_values("k").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_pair(cs: CrossSection, K: float = 0.90) -> pd.DataFrame:
    """Select the two OTM put quotes that bracket the target strike *K*.

    The **pair** consists of:

    * the OTM put with the largest ``k`` that is still below *K*
      (closest put below the target), and
    * the OTM put with the smallest ``k`` that is above *K* but still
      OTM (i.e., ``K < k < 1``).

    Parameters
    ----------
    cs:
        Cross-section containing the normalised quote table.
    K:
        Target normalised strike (default ``0.90``).

    Returns
    -------
    pd.DataFrame
        0, 1, or 2 rows with columns ``['k', 'c_lo', 'c_hi', 'opt_type']``.
    """
    puts = _otm_puts(cs)

    if puts.empty:
        logger.warning(
            "[build_pair] %s h=%d: no OTM puts available.",
            cs.trade_date, cs.horizon,
        )
        return _empty_quotes()

    below = puts[puts["k"] < K]
    above = puts[(puts["k"] > K) & (puts["k"] < 1.0)]

    parts: list[pd.DataFrame] = []

    if below.empty:
        logger.warning(
            "[build_pair] %s h=%d: no OTM put below K=%.4f.",
            cs.trade_date, cs.horizon, K,
        )
    else:
        # Closest below K → largest k
        parts.append(below.iloc[[-1]])

    if above.empty:
        logger.warning(
            "[build_pair] %s h=%d: no OTM put above K=%.4f (and below 1).",
            cs.trade_date, cs.horizon, K,
        )
    else:
        # Closest above K → smallest k
        parts.append(above.iloc[[0]])

    if not parts:
        return _empty_quotes()

    return _dedup_sort(pd.concat(parts, ignore_index=True))


def build_local(cs: CrossSection, K: float = 0.90) -> pd.DataFrame:
    """Select the pair plus the six nearest OTM puts to *K*.

    Constructs the **local** set as the union of:

    * the pair (see :func:`build_pair`), and
    * the six OTM put quotes with the smallest ``|k - K|``.

    Duplicates are removed before returning.  If fewer than six OTM
    puts exist the full OTM put set is used instead.

    Parameters
    ----------
    cs:
        Cross-section containing the normalised quote table.
    K:
        Target normalised strike (default ``0.90``).

    Returns
    -------
    pd.DataFrame
        Up to 8 rows sorted by ``k``.
    """
    puts = _otm_puts(cs)

    if puts.empty:
        logger.warning(
            "[build_local] %s h=%d: no OTM puts available.",
            cs.trade_date, cs.horizon,
        )
        return _empty_quotes()

    # Six nearest OTM puts by distance to K
    n_nearest = min(6, len(puts))
    nearest_idx = (puts["k"] - K).abs().nsmallest(n_nearest).index
    six_nearest = puts.loc[nearest_idx]

    pair = build_pair(cs, K=K)

    combined = pd.concat([pair, six_nearest], ignore_index=True)
    return _dedup_sort(combined)


def build_wing(cs: CrossSection) -> pd.DataFrame:
    """Return all OTM puts, sorted by ``k``.

    The **wing** set is the full left wing of the implied volatility surface
    represented in the cross-section.

    Parameters
    ----------
    cs:
        Cross-section containing the normalised quote table.

    Returns
    -------
    pd.DataFrame
        All OTM put quotes sorted by ``k`` ascending.
    """
    puts = _otm_puts(cs)

    if puts.empty:
        logger.warning(
            "[build_wing] %s h=%d: no OTM puts available.",
            cs.trade_date, cs.horizon,
        )

    return puts.reset_index(drop=True)


def build_full_otm(cs: CrossSection) -> pd.DataFrame:
    """Return all OTM puts and OTM calls, sorted by ``k``.

    The **full_otm** set spans the complete OTM surface: left wing (puts,
    ``k < 1``) and right wing (calls, ``k > 1``).

    Parameters
    ----------
    cs:
        Cross-section containing the normalised quote table.

    Returns
    -------
    pd.DataFrame
        All OTM quotes sorted by ``k`` ascending.
    """
    puts = _otm_puts(cs)
    calls = _otm_calls(cs)

    if puts.empty:
        logger.warning(
            "[build_full_otm] %s h=%d: no OTM puts available.",
            cs.trade_date, cs.horizon,
        )
    if calls.empty:
        logger.warning(
            "[build_full_otm] %s h=%d: no OTM calls available.",
            cs.trade_date, cs.horizon,
        )

    combined = pd.concat([puts, calls], ignore_index=True)
    return combined.sort_values("k").reset_index(drop=True)


def build_all_quote_sets(
    cs: CrossSection,
    K: float = 0.90,
) -> dict[str, pd.DataFrame]:
    """Build all four quote sets for a cross-section in one call.

    Parameters
    ----------
    cs:
        Cross-section to process.
    K:
        Target normalised strike used for ``pair`` and ``local``
        (default ``0.90``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping with keys ``'pair'``, ``'local'``, ``'wing'``,
        ``'full_otm'``.
    """
    return {
        "pair": build_pair(cs, K=K),
        "local": build_local(cs, K=K),
        "wing": build_wing(cs),
        "full_otm": build_full_otm(cs),
    }
