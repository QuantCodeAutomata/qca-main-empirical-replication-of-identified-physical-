"""Synthetic SPX option data generator mimicking OptionMetrics end-of-day data.

Generates OTM put and call quotes using Black-Scholes pricing with a vol smile
(put skew) for all Wednesdays from 2013-01-02 to 2023-08-31 across three
target horizons: 23, 30, and 37 days to maturity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes primitives
# ---------------------------------------------------------------------------

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_S0 = 1500.0
_GROWTH = 0.07          # annual drift for spot
_R = 0.02               # risk-free rate
_BASE_VOL = 0.18        # ATM implied vol
_SKEW = -0.5            # d(IV)/d(log-moneyness)
_SIGMA_FLOOR = 0.005    # minimum implied vol to keep BS well-behaved
_MIN_ASK = 0.01         # minimum ask price (cents floor) to avoid zero-ask from norm.cdf underflow
_HORIZONS = [23, 30, 37]
_MONEYNESS_GRID = np.arange(0.70, 1.301, 0.02)   # 0.70, 0.72, …, 1.30


def _spot_at(trade_date: pd.Timestamp, start: pd.Timestamp) -> float:
    years = (trade_date - start).days / 365.0
    return _S0 * np.exp(_GROWTH * years)


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic SPX options dataset using Black-Scholes pricing.

    Parameters
    ----------
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: trade_date, maturity_date, days_to_maturity, option_type,
        strike, bid, ask, volume, discount_factor, forward_price.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2013-01-02")
    end = pd.Timestamp("2023-08-31")
    wednesdays = pd.date_range(start, end, freq="W-WED")

    rows = []

    for trade_date in wednesdays:
        S = _spot_at(trade_date, start)

        for horizon in _HORIZONS:
            T = horizon / 365.0
            D = np.exp(-_R * T)
            F = S * np.exp(_R * T)
            maturity_date = trade_date + pd.Timedelta(days=horizon)

            strikes = _MONEYNESS_GRID * F

            for K in strikes:
                log_m = np.log(K / F)                           # log(k), k = K/F
                sigma = max(_BASE_VOL + _SKEW * log_m, _SIGMA_FLOOR)
                spread = max(0.03, 0.05 * abs(log_m))

                # ----- call -----
                call_mid = bs_call(S, K, T, _R, sigma)
                call_bid = max(call_mid * (1.0 - spread), 0.0)
                call_ask = max(call_mid * (1.0 + spread), _MIN_ASK)
                call_vol = int(rng.integers(100, 1001))

                # ----- put -----
                put_mid = bs_put(S, K, T, _R, sigma)
                put_bid = max(put_mid * (1.0 - spread), 0.0)
                put_ask = max(put_mid * (1.0 + spread), _MIN_ASK)
                put_vol = int(rng.integers(100, 1001))

                base = dict(
                    maturity_date=maturity_date.strftime("%Y-%m-%d"),
                    days_to_maturity=horizon,
                    strike=round(K, 6),
                    discount_factor=round(D, 8),
                    forward_price=round(F, 6),
                    trade_date=trade_date.strftime("%Y-%m-%d"),
                )

                rows.append({**base,
                             "option_type": "C",
                             "bid": round(call_bid, 4),
                             "ask": round(call_ask, 4),
                             "volume": call_vol})
                rows.append({**base,
                             "option_type": "P",
                             "bid": round(put_bid, 4),
                             "ask": round(put_ask, 4),
                             "volume": put_vol})

    df = pd.DataFrame(rows, columns=[
        "trade_date", "maturity_date", "days_to_maturity",
        "option_type", "strike", "bid", "ask", "volume",
        "discount_factor", "forward_price",
    ])
    return df


def load_or_generate(path: str = "data/synthetic_spx_options.parquet") -> pd.DataFrame:
    """Load the synthetic dataset from *path* if it exists, otherwise generate,
    persist, and return it.

    Parameters
    ----------
    path:
        Parquet file path.  Relative paths are resolved from the current
        working directory.

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)

    df = generate_synthetic_data()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df
