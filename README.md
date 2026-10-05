# Physical Crash-Risk Identification from SPX Options

Replication and extension of identified-set analysis for physical crash probability,
unconditional shortfall, and conditional depth from finite SPX option bid-ask quotes
under a power-utility pricing kernel.

## Overview

This project implements the exact SOCP-based partial-identification methodology for
measuring crash risk from options data.  The key contribution is that finite bid-ask
quotes only *partially* identify physical crash risk even after imposing a power-utility
kernel restriction — a convex identified *set* rather than a point estimate results.

### Key parameters (benchmark specification)
- **γ = 2** — power-utility pricing kernel  
- **K = 0.90** — crash threshold (normalized index level)  
- **B = 5** — upper support cap for normalized index  
- **Horizons:** 23, 30, 37 calendar days  

---

## Repository structure

```
repo/
├── data/
│   ├── data_loader.py           # Load, filter, normalize option cross sections
│   └── synthetic_generator.py  # Generate synthetic SPX option data for testing
│
├── src/                         # Core implementation
│   ├── normalize.py             # DF/F normalization, parity conversion, OTM split
│   ├── quote_sets.py            # Pair / local / wing / full-OTM quote-set builder
│   ├── regularity.py            # Assumption 3.3 feasibility verification
│   ├── socp_builder.py          # Shared SOCP model infrastructure
│   ├── socp_q_l.py              # SOCP for crash probability q and shortfall l
│   ├── socp_s.py                # SOCP for conditional depth s
│   ├── frontier.py              # Joint (q,l) frontier via support-function SOCP
│   ├── bootstrap.py             # Circular block bootstrap for width-reduction CIs
│   ├── robustness.py            # Finite-B and unbounded-tail robustness
│   └── quote_manipulations.py  # Spread doubling, strike thinning, reverse deletion
│
├── exp/                         # Experiment scripts
│   ├── exp1_benchmark.py        # Main identified intervals — Table 1, Figures 1–2
│   ├── exp2_frontier.py         # Joint frontier reconstruction — Figures 4–6, Table 5
│   ├── exp3_robustness_B.py     # Support-cap and unbounded-tail robustness — Figure 3
│   ├── exp4_kernel_threshold.py # γ=1 vs γ=2 and K=0.85 robustness — Tables 2–3
│   └── exp5_perturbation.py     # Quote perturbation and attribution — Tables 4, 6
│
├── tests/                       # pytest test suite (92 tests)
│   ├── test_normalize.py
│   ├── test_quote_sets.py
│   ├── test_socp_builder.py
│   ├── test_socp_solvers.py
│   └── test_bootstrap.py
│
├── results/                     # Saved outputs
│   ├── RESULTS.md               # Summary of all experiments and findings
│   ├── exp1_table1.csv          # Median intervals by horizon × quote-set
│   ├── exp1_bootstrap_reduction.csv
│   ├── exp2_gap_diagnostics.csv # Frontier approximation quality
│   ├── exp3_finite_B_bounds.csv
│   ├── exp3_unbounded_relaxation.csv
│   ├── exp4_table2_gamma.csv
│   ├── exp4_table3_threshold.csv
│   ├── exp5_spread_thinning.csv
│   ├── exp5_deletion_widening.csv
│   └── *.png                    # All figures
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a working CVXPY installation with at least one conic solver
(CLARABEL is used by default; MOSEK or SCS are alternatives).

---

## Running experiments

```bash
# (Optional) regenerate synthetic data
python data/synthetic_generator.py

# Experiment 1 — benchmark identified intervals
python exp/exp1_benchmark.py

# Experiment 2 — joint frontier reconstruction
python exp/exp2_frontier.py

# Experiment 3 — finite-B and unbounded-tail robustness
python exp/exp3_robustness_B.py

# Experiment 4 — kernel and threshold robustness
python exp/exp4_kernel_threshold.py

# Experiment 5 — quote perturbation and attribution
python exp/exp5_perturbation.py
```

Results are written to `results/`.

---

## Running tests

```bash
pytest tests/ -v
```

All 92 tests should pass in under 60 seconds.

---

## Data note

The original paper uses proprietary OptionMetrics end-of-day SPX option data.  This
repository uses **synthetic** option cross sections generated via a Black-Scholes /
jump-diffusion model that mimics OptionMetrics data structure.  Real OptionMetrics data
can be dropped in by replacing `data/synthetic_spx_options.parquet` with a parquet
file with the same schema (columns: `trade_date`, `maturity_date`, `days_to_maturity`,
`option_type`, `strike`, `bid`, `ask`, `volume`, `discount_factor`, `forward_price`).

---

## Methodology

The identified set M(I) consists of all risk-neutral measures Q on [0, B] satisfying:

1. **Risk-neutral condition:** E_Q[X] = 1  
2. **Quote inequalities:** c̲ᵢ ≤ E_Q[(X−κᵢ)⁺] ≤ c̄ᵢ for all retained OTM quotes  
3. **Support:** Q supported on [0, B]

Under the power-utility kernel (dP/dQ ∝ X^γ), physical crash-risk measures are:

- **q(Q)** = A₂(Q) / M(Q) — crash probability  
- **l(Q)** = (K·A₂(Q) − A₃(Q)) / M(Q) — unconditional shortfall  
- **s(Q)** = l(Q)/q(Q) — conditional depth  

where Aᵣ(Q) = E_Q[X^r · 1{X ≤ K}] and M(Q) = E_Q[X²].

Minimizing and maximizing each measure over M(I) yields the identified interval via an
exact finite-dimensional SOCP based on the Charnes-Cooper scaling trick.

---

## License

MIT
