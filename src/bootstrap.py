"""Circular block bootstrap inference for identified-set width series.

Provides two estimators suited for weakly-dependent time series of
identified-set widths:

* :func:`circular_block_bootstrap` — percentile confidence interval for the
  median width of a single series.
* :func:`bootstrap_width_reduction` — percentile CI for the median
  *proportional* width reduction ``1 - w_finer / w_coarser`` between two
  nested quote sets, using a paired circular block bootstrap that preserves
  cross-series dependence.

Algorithm (circular block bootstrap)
-------------------------------------
Given *n* clean observations arranged on a circle, draw
``n_blocks = ceil(n / block_length)`` starting positions uniformly from
``{0, …, n-1}``.  Each block wraps around via modular indexing.  Concatenate
blocks, trim to *n* observations, compute the statistic of interest.  Repeat
``n_bootstrap`` times and form a percentile confidence interval.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def circular_block_bootstrap(
    widths: np.ndarray,
    n_bootstrap: int = 2000,
    block_length: int = 8,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Circular block bootstrap confidence interval for the median width.

    Parameters
    ----------
    widths:
        1-D array of identified-set widths (may contain NaN values, which
        are dropped before bootstrapping).
    n_bootstrap:
        Number of bootstrap replications.
    block_length:
        Length of each contiguous block drawn from the circular series.
    alpha:
        Nominal coverage is ``1 - alpha``; the CI uses the
        ``[alpha/2, 1-alpha/2]`` percentiles of the bootstrap distribution.
    seed:
        Integer seed for reproducibility.

    Returns
    -------
    dict
        ``{'median': float, 'ci_lower': float, 'ci_upper': float,
        'bootstrap_medians': np.ndarray}``
    """
    w = np.asarray(widths, dtype=float)
    w = w[~np.isnan(w)]
    n = len(w)

    if n == 0:
        logger.warning("circular_block_bootstrap: no valid (non-NaN) observations.")
        return {
            "median": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "bootstrap_medians": np.array([], dtype=float),
        }

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_length)
    boot_medians = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        # Draw block starting positions uniformly from {0, ..., n-1}
        starts = rng.integers(0, n, size=n_blocks)
        # Build index array with wrap-around, then trim to n
        indices = np.concatenate(
            [np.arange(s, s + block_length) % n for s in starts]
        )[:n]
        boot_medians[b] = np.median(w[indices])

    ci_lo = float(np.percentile(boot_medians, 100.0 * alpha / 2.0))
    ci_hi = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "median": float(np.median(w)),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "bootstrap_medians": boot_medians,
    }


def bootstrap_width_reduction(
    widths_coarser: np.ndarray,
    widths_finer: np.ndarray,
    n_bootstrap: int = 2000,
    block_length: int = 8,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the median proportional width reduction.

    Estimates the median of ``r_t = 1 - w_finer_t / w_coarser_t`` across
    time periods *t*.  A *paired* circular block bootstrap is used so that
    the same block indices are applied to both width series, preserving
    cross-series dependence.

    Time periods where either series is NaN or ``w_coarser`` is zero are
    excluded from both series before bootstrapping.

    Parameters
    ----------
    widths_coarser:
        Width series for the coarser (larger) identified set.
    widths_finer:
        Width series for the finer (smaller) identified set; must have the
        same length as ``widths_coarser``.
    n_bootstrap:
        Number of bootstrap replications.
    block_length:
        Circular block length.
    alpha:
        Nominal coverage ``1 - alpha``.
    seed:
        Integer seed for reproducibility.

    Returns
    -------
    dict
        ``{'median_reduction': float, 'ci_lower': float, 'ci_upper': float}``
    """
    wc = np.asarray(widths_coarser, dtype=float)
    wf = np.asarray(widths_finer, dtype=float)

    if wc.shape != wf.shape:
        raise ValueError(
            f"widths_coarser and widths_finer must have the same shape; "
            f"got {wc.shape} vs {wf.shape}."
        )

    # Keep only time periods where both series are valid and denominator != 0
    valid = ~(np.isnan(wc) | np.isnan(wf) | (wc == 0.0))
    wc, wf = wc[valid], wf[valid]
    n = len(wc)

    if n == 0:
        logger.warning(
            "bootstrap_width_reduction: no valid paired observations after "
            "dropping NaNs and zero-denominator entries."
        )
        return {"median_reduction": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    reductions = 1.0 - wf / wc
    median_red = float(np.median(reductions))

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_length)
    boot_medians = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, s + block_length) % n for s in starts]
        )[:n]
        # Paired: same index array applied to both series
        boot_red = 1.0 - wf[indices] / wc[indices]
        boot_medians[b] = np.median(boot_red)

    ci_lo = float(np.percentile(boot_medians, 100.0 * alpha / 2.0))
    ci_hi = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "median_reduction": median_red,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }
