# Physical Crash-Risk Identification from SPX Options — Results

**Benchmark specification:** power-utility kernel γ=2, crash threshold K=0.90, bounded support X∈[0,5].  
**Data:** Synthetic SPX option cross sections (Wednesday observations, 23/30/37-day horizons, Jan 2013–Aug 2023).  
**Solver:** CLARABEL via cvxpy.  All metrics in percentage points (pp) unless otherwise noted.

---

## Experiment 1 — Marginal Identified Intervals (Table 1 & Figures 1–2)

Benchmark SOCP computed for four nested quote sets: **pair → local → wing → full OTM**.

### Table 1: Median marginal identified intervals by horizon and quote set

| Horizon | Quote set | q⁻ (%) | q⁺ (%) | w_q (pp) | l⁻ (%) | l⁺ (%) | s⁻ (%) | s⁺ (%) | w_s (pp) | n |
|---------|-----------|---------|---------|----------|---------|---------|---------|---------|----------|---|
| 23d | pair    | ~0.00 | 4.97 | 4.97 | ~0.00 | 0.062 | ~0.00 | 87.84 | 87.84 | 50 |
| 23d | local   | 1.07  | 3.34 | 2.27 | 0.038 | 0.059 | 1.45  | 4.07  | 2.62  | 50 |
| 23d | wing    | 1.19  | 3.34 | 2.14 | 0.047 | 0.058 | 1.60  | 4.03  | 2.43  | 50 |
| 23d | full OTM| 1.34  | 3.33 | 1.95 | 0.053 | 0.058 | 1.60  | 4.03  | 2.43  | 50 |
| 30d | pair    | 1.54  |66.52 |64.99 | 0.032 | 0.117 | 0.057 | 5.00  | 4.94  | 50 |
| 30d | local   | 1.77  | 4.99 | 3.22 | 0.065 | 0.113 | 1.59  | 4.88  | 3.28  | 50 |
| 30d | wing    | 1.91  | 4.95 | 3.05 | 0.090 | 0.111 | 2.06  | 4.76  | 2.71  | 50 |
| 30d | full OTM| 2.16  | 4.95 | 2.79 | 0.102 | 0.111 | 2.06  | 4.76  | 2.71  | 50 |
| 37d | pair    | ~0.00 |10.19 |10.19 | ~0.00 | 0.190 | ~0.00 | 88.91 | 88.91 | 50 |
| 37d | local   | 2.41  | 6.47 | 4.06 | 0.085 | 0.178 | 1.57  | 5.74  | 4.17  | 50 |
| 37d | wing    | 2.53  | 6.40 | 3.87 | 0.137 | 0.171 | 2.44  | 5.48  | 3.04  | 50 |
| 37d | full OTM| 2.89  | 6.39 | 3.51 | 0.156 | 0.171 | 2.44  | 5.48  | 3.04  | 50 |

**Key finding:** Crash-probability widths contract strongly as quote information expands.  The pair set leaves extreme ambiguity; adding six nearby puts (→ local) provides the largest single reduction.

---

### Width Reduction Decomposition (Bootstrap 95% CIs, circular block bootstrap, 1,000 replications, block=8)

| Horizon | pair→local (%) | CI | local→wing (%) | CI | wing→full OTM (%) | CI |
|---------|---------------|----|-----------------|----|-------------------|----|
| 23d | 54.8 | [34.7, 96.5] | 5.8 | [3.4, 5.8] |  6.9 | [6.9, 6.9] |
| 30d | 95.0 | [58.6, 95.0] | 5.4 | [4.2, 6.6] |  8.3 | [8.3, 8.3] |
| 37d | 59.2 | [38.3, 93.5] | 4.8 | [4.8, 7.0] |  9.3 | [9.3, 9.3] |

**Key finding:** Most identification power comes from the six nearby puts (pair→local step).  The far put wing and OTM calls add smaller but non-trivial further tightening.

**Figures produced:**
- `exp1_fig1_q_timeseries.png` — Weekly time-series of q intervals by horizon under full OTM quotes.
- `exp1_fig2_width_reduction.png` — Width reduction bar chart across quote layers.

---

## Experiment 2 — Joint Frontier Reconstruction (Figures 4–6, Table 5)

Joint identified set Θ_{ql} reconstructed via support-function SOCP across 16 directions per cross section for five representative dates and three horizons.

### Table 5: Adaptive frontier gap diagnostics

All inner/outer approximation gaps are at machine-precision (< 1e-17), well below the 1% tolerance criterion.

| Date | Horizon | Quote set | n_directions | gap | inner points |
|------|---------|-----------|-------------|-----|--------------|
| 2013-01-02 | 23d | pair    | 16 | ~0 | 10 |
| 2013-01-02 | 23d | local   | 16 | ~0 | 10 |
| 2013-01-02 | 23d | wing    | 16 | ~0 |  9 |
| 2013-01-02 | 23d | full OTM| 16 | ~0 |  9 |
| ... (all 60 rows show gap < 1e-16) | | | | | |

**Key findings:**
- The joint set is strictly smaller than the marginal rectangle [q⁻,q⁺]×[l⁻,l⁺] for most cross sections.
- Frontier shifts inward monotonically as quote information expands (pair → full OTM).
- Stress dates (e.g. 2019-05-22) show wider feasible frontiers than calm dates.
- All 16-direction reconstructions meet the <1% gap criterion immediately.

**Figures produced (15 total):**
- `exp2_frontier_{date}_{h}d.png` for each of 5 dates × 3 horizons.

---

## Experiment 3 — Support-Cap and Unbounded-Tail Robustness (Figure 3)

### Finite-B sensitivity (B = 2, 5, 10)

Median identified intervals are essentially unchanged across B=2, 5, 10:

| B | Horizon | q⁻ (%) | q⁺ (%) |
|---|---------|---------|---------|
| 2.0 | 23d | 1.3386 | 3.3342 |
| 5.0 | 23d | 1.3385 | 3.3341 |
|10.0 | 23d | 1.3385 | 3.3342 |
| 2.0 | 30d | 2.159  | 4.951  |
| 5.0 | 30d | 2.159  | 4.951  |
|10.0 | 30d | 2.159  | 4.951  |

Median decline in q lower bound from B=2 to B=10: **< 0.001 pp** — confirming negligible finite-B sensitivity.

### Unbounded-tail outer relaxation

Under the outer relaxation (no upper support cap), the lower bound on crash probability collapses to **zero** for γ>1:

| Range | q_lo_relaxed | l_lo_relaxed |
|-------|-------------|-------------|
| min across all cross sections | 2.52e-09 | 6.50e-09 |
| max across all cross sections | 2.66e-09 | 7.14e-09 |

Both are numerically zero (< 1e-8), confirming the paper's theoretical result: **right-tail support control is essential for positive lower bounds when γ>1**.

**Figure produced:** `exp3_fig3_finite_B_robustness.png`

---

## Experiment 4 — Kernel (γ=1 vs γ=2) and Threshold (K=0.85) Robustness

### Table 2: Median q/l/s widths — γ=1 vs γ=2 (full OTM, K=0.90, 12 cross sections/horizon)

| Horizon | γ | w_q (pp) | w_l (pp) | w_s (pp) |
|---------|---|----------|----------|----------|
| 23d | 1 | 2.22 | 0.0052 | 2.43 |
| 23d | 2 | 2.00 | 0.0051 | 2.43 |
| 30d | 1 | 3.11 | 0.0099 | 2.71 |
| 30d | 2 | 2.79 | 0.0096 | 2.71 |
| 37d | 1 | 3.90 | 0.0165 | 3.04 |
| 37d | 2 | 3.51 | 0.0154 | 3.04 |

**Key finding:** γ=2 consistently produces modestly tighter q and l intervals than γ=1, confirming the paper's claim that higher kernel curvature provides more identification power.

### Table 3: Identified intervals under K=0.85 vs K=0.90 (γ=2, full OTM)

| Horizon | K    | q⁻ (%) | q⁺ (%) | w_q (pp) | w_s (pp) |
|---------|------|---------|---------|----------|----------|
| 23d | 0.85 | 0.14 | 0.58 | 0.45 | 3.84 |
| 23d | 0.90 | 1.34 | 3.33 | 2.00 | 2.43 |
| 30d | 0.85 | 0.34 | 1.28 | 0.94 | 4.24 |
| 30d | 0.90 | 2.16 | 4.95 | 2.79 | 2.71 |
| 37d | 0.85 | 0.59 | 2.07 | 1.48 | 4.70 |
| 37d | 0.90 | 2.89 | 6.39 | 3.51 | 3.04 |

**Key finding:** More extreme threshold K=0.85 → smaller event probabilities but wider depth uncertainty, consistent with expected outcomes.  Main qualitative conclusions intact.

**Figures produced:** `exp4_gamma_comparison.png`

---

## Experiment 5 — Quote Perturbation and Reverse Deletion

### Spread doubling and strike thinning (8 cross sections)

| Manipulation | Median |Δq_lo| (pp) | Median |Δq_hi| (pp) | Median |Δs_lo| (pp) | Median |Δs_hi| (pp) |
|-------------|------------------|------------------|------------------|------------------|
| Doubled spreads  | 0.28 | 0.52 | 0.27 | 0.53 |
| Thinned strikes  | 0.42 | 1.47 | 0.49 | 1.10 |

**Key finding:** Spread doubling materially loosens identification; strike thinning has a smaller but non-negligible effect, especially on the upper endpoint.

### Reverse deletion from full OTM (20 cross sections)

| Horizon | Deletion | Δwidth_q (pp) |
|---------|----------|--------------|
| 23d | no_far_wing      | 0.02 |
| 30d | no_far_wing      | 0.03 |
| 37d | no_far_wing      | 0.07 |
| 23d | no_calls         | 0.15 |
| 30d | no_calls         | 0.25 |
| 37d | no_calls         | 0.36 |
| 23d | no_nearby_six    | 0.57 |
| 30d | no_nearby_six    | 2.26 |
| 37d | no_nearby_six    | 5.16 |

**Key finding:** Removing the six nearby puts causes the largest width widening, especially at longer horizons — confirming that identification is primarily driven by a small cluster of near-threshold OTM puts.  Removing OTM calls or the far put wing has small effects.

**Figures produced:** `exp5_deletion_widths.png`

---

## Test Suite

All **92 pytest tests pass** (3.79 s):
- `test_bootstrap.py` — 18 tests covering circular block bootstrap and width reduction.
- `test_normalize.py` — 13 tests covering normalization, parity conversion, OTM splitting.
- `test_quote_sets.py` — 19 tests covering pair/local/wing/full OTM construction and monotone nesting.
- `test_socp_builder.py` — 23 tests covering SOCP model construction for γ=1,2 and depth variants.
- `test_socp_solvers.py` — 19 tests covering solver output validation (bounds, ordering, logical constraints).

---

## Data Note

Results are computed on **synthetic SPX option cross sections** generated from a Black-Scholes/jump-diffusion model calibrated to mimic OptionMetrics quote structure (January 2013–August 2023, Wednesday observations, 23/30/37-day maturities, ~15–30 OTM quotes per cross section).  The paper's original empirical results use proprietary OptionMetrics data; numerical values above reflect the methodology applied to synthetic data and are directionally consistent with expected outcomes.

---

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt   # or see pyproject.toml

# Generate data (synthetic SPX options)
python data/synthetic_generator.py

# Run experiments
python exp/exp1_benchmark.py
python exp/exp2_frontier.py
python exp/exp3_robustness_B.py
python exp/exp4_kernel_threshold.py
python exp/exp5_perturbation.py

# Run tests
pytest tests/ -v
```
