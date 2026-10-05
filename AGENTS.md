# Project: SPX Physical Crash-Risk Identification

## Overview
Quantitative finance project estimating physical crash-risk from SPX options.
Uses Black-Scholes synthetic data (mimicking OptionMetrics) with forward-normalised
call-price quote sets fed into crash-risk estimators.

## Package Structure
```
data/
  __init__.py
  synthetic_generator.py   # BS data generation + load_or_generate
  data_loader.py           # CrossSection dataclass + load_cross_sections + filter_otm
src/
  __init__.py
  normalize.py             # normalize_quotes, parity_convert, filter_valid_quotes, split_otm
  quote_sets.py            # build_pair, build_local, build_wing, build_full_otm, build_all_quote_sets
exp/                       # experiment scripts
results/
tests/
```

## Key Design Decisions

### Normalisation convention
All option prices are expressed in **forward-normalised call-price space**:
  `c = option_price / (D * F)`   where D=exp(-rT), F=S*exp(rT)

Puts are immediately converted to call-equivalent via put-call parity:
  `c_put_equiv = p/(D*F) + 1 - k`   where k = K/F

This means `CrossSection.quotes` always contains call-equivalent prices for both types.

### Two-layer normalisation (data_loader vs src/normalize)
- `data_loader._normalise_all` applies full normalisation (including parity for puts) to the *whole* raw DataFrame at once — efficient pipeline path.
- `src.normalize.*` provides composable standalone functions: `normalize_quotes` does raw→c conversion WITHOUT parity; `parity_convert` adds parity separately. Useful for research/experimentation.

### Synthetic data MIN_ASK floor
`_MIN_ASK = 0.01` is applied to ask prices in `synthetic_generator.py` to prevent zero-ask values that arise from `scipy.stats.norm.cdf` underflow on deep OTM options (K up to 1.30·F with σ≈0.005). Bid remains `max(mid*(1−s), 0)`, which can be 0 for negligible-value options.

### Wednesday frequency
The generator uses `pd.date_range(freq="W-WED")`. `data_loader._keep_wednesdays` uses `dayofweek == 2` to filter real data to Wednesdays only.

### Maturity matching (±3 days)
`load_cross_sections` matches each horizon to any maturity within `|DTM − horizon| ≤ 3` days, then picks the closest. This accommodates real OptionMetrics data where exact horizon maturities may not exist.

## Crash-Risk Analysis Modules (src/)

Five modules support crash-risk identified-set experiments; all depend on
`src.socp_builder.SOCPBuilder` (not yet created) except `bootstrap.py` and
`quote_manipulations.py`.

### Expected SOCPBuilder interface (assumed by all SOCP modules)
```python
class SOCPBuilder:
    # Attributes (set in __init__)
    grid_lo:  np.ndarray   # lower bounds of return-grid cells (n_cells,)
    grid_hi:  np.ndarray   # upper bounds of return-grid cells (n_cells,)
    grid_mid: np.ndarray   # cell midpoints
    n_cells:  int

    # Methods
    admissibility_constraints(mass: cp.Variable) -> List[cp.Constraint]
        # Non-negativity, sum=1, martingale, gamma-moment constraints

    quote_constraints(mass: cp.Variable, slack=0.0) -> List[cp.Constraint]
        # quote: price >= c_lo + slack, price <= c_hi - slack
        # slack can be 0, a float, or a cp.Variable (for regularity check)

    crash_prob_expr(mass)     -> cp.Expression  # Q(R < K)
    crash_log_loss_expr(mass) -> cp.Expression  # E_Q[log(R) · 1(R<K)]
    crash_shortfall_expr(mass)-> cp.Expression  # E_Q[(K-R) · 1(R<K)]
```

### src/regularity.py — `verify_regularity(cs, quotes_df, ...)`
Maximises slack `ε` s.t. tightened quote constraints and positive mass on
`U_minus=(0.3, 0.5)` and `U_plus=(1.2, 1.8)`.
Returns `{'regular': bool, 'slack': float, 'status': str}`.

### src/bootstrap.py — pure NumPy, no SOCP
- `circular_block_bootstrap(widths, ...)` → CI for median width.
- `bootstrap_width_reduction(widths_coarser, widths_finer, ...)` → CI for
  median proportional reduction `1 - w_finer/w_coarser` via **paired** CBB.
  Raises `ValueError` on shape mismatch. Strips NaN and zero-denominator
  entries before resampling.

### src/frontier.py — `FrontierSolver` class
`solve_support_function(direction)` solves `max u_q·q + u_l·l` via SOCP.
`build_frontier(n_initial, max_directions, gap_tol)` adaptively bisects the
direction with the largest outer–inner gap. Returns convex-hull vertices as
`inner_points`. Gap formula: `h(u) − max_j u·(q_j, l_j)`.

### src/robustness.py
- `compute_bounds_varying_B(cs, quotes_df, B_values, ...)` → DataFrame with
  columns `[B, q_lo, q_hi, l_lo, l_hi, s_lo, s_hi]`. Runs 6 SOCPs per B.
- `compute_unbounded_relaxation(cs, quotes_df, B_relaxed=50, ...)` → dict
  with `q_lo_relaxed`, `l_lo_relaxed`, `status`.

### src/quote_manipulations.py — pure pandas, no SOCP
All functions take/return `['k', 'c_lo', 'c_hi', 'opt_type']` DataFrames,
sorted ascending by `k`, with index reset.
- `double_spreads`: doubles spread around midpoint, clips c_lo ≥ 0.
- `thin_strikes`: keeps every other strike for P and C separately.
- `remove_far_put_wing(quotes_df, local_df)`: keeps local puts + OTM calls.
- `remove_otm_calls(quotes_df)`: drops all C rows, returns puts only.
- `remove_nearby_six_puts(quotes_df, pair_df, local_df)`: drops puts in
  `local_k − pair_k` from quotes_df. Matching by exact `k` float value
  (safe: same CrossSection source).

## Dependencies
- numpy, pandas, scipy (core)
- cvxpy ≥ 1.7 + CLARABEL solver (for regularity, frontier, robustness)
- pyarrow (required for parquet read/write in load_or_generate)

## Running a quick sanity check
```python
import sys; sys.path.insert(0, '.')
from data.synthetic_generator import generate_synthetic_data
from data.data_loader import load_cross_sections
from src.quote_sets import build_all_quote_sets

df = generate_synthetic_data()       # 103,602 rows, 557 Wednesdays
css = load_cross_sections()          # 1,671 CrossSections (557 × 3 horizons)
sets = build_all_quote_sets(css[0])  # {'pair', 'local', 'wing', 'full_otm'}
```
