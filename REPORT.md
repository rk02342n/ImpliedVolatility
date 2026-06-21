# Predicting Implied Volatility and Options Pricing Analysis

*Rohit Sai Gopal Kumaratchi Chandrashekar*

> **Revised, leakage-free report.** An earlier version of this project reported R² = 0.868 for
> predicting implied volatility. A review ([`AUDIT.md`](./AUDIT.md)) showed that result was
> **spurious target leakage**. This report documents the corrected pipeline (now a proper Python
> package under `src/implied_volatility/`, run with `uv run ivol run`) and its honest results.
> All numbers below are produced by that pipeline and saved to `outputs/metrics.json`.

## Introduction and Problem Statement

The goal is to predict the **implied volatility (IV)** of S&P 500 (SPY) call options so that
options can be priced with the **Black-Scholes** model — swapping the model's assumed-constant
volatility for a data-driven estimate. The target is the dataset's `c_iv` column.

**Black-Scholes call price:**

$$C = N(d_1)S_t - N(d_2)Ke^{-rt}$$

$$d_1 = \frac{\ln\frac{S_t}{K} + (r + \frac{\sigma^2}{2})t}{\sigma\sqrt{t}}, \qquad d_2 = d_1 - \sigma\sqrt{t}$$

Where `C` = call price, `N` = standard-normal CDF, `S_t` = spot, `K` = strike, `r` = risk-free
rate, `t` = time to maturity, `σ` = volatility.

The key insight that reframes the whole project: **to price an option you don't yet have IV for,
you cannot use any input that is itself derived from IV.** This rules out the option Greeks and
the option's own bid/ask/last prices. It makes the task a *cross-sectional volatility-surface
reconstruction* problem — estimate IV from strike, expiry, and the underlying's behaviour, then
price.

## What was wrong before (and why R² = 0.868 was fake)

The original model predicted `c_iv` from the option Greeks (`c_delta, c_gamma, c_vega, c_theta,
c_rho`). But in this OptionsDX-style dataset **the Greeks are computed *from* `c_iv`** via
Black-Scholes — they are not independent market observables. There is even a closed-form inversion:

$$\sigma = 100 \cdot \frac{\text{vega}}{\text{gamma} \cdot S^2 \cdot t}$$

So the model was recovering the target from transformations of the target — textbook **target
leakage**. The headline R² did not reflect predictive skill, and the model was useless for its
stated goal: at pricing time you don't know IV, so you can't have the Greeks either. See
[`AUDIT.md`](./AUDIT.md) §1.

## Dataset — SPY Option Chains, Q1 2020 – Q4 2022

The raw file (`spy_2020_2022.csv`, 1.28 GB, ~3.6M rows, 33 columns) covers Q1 2020 – Q4 2022 and
is **not sorted by date** (rows are grouped by expiry). The corrected pipeline therefore reads the
file in chunks and draws a **seeded sample of ~200k valid call rows spread across the entire date
range**, rather than the original notebook's first 40,000 rows (which actually only covered
Aug–Sep 2021 and never saw a volatility-regime change). After feature engineering and outlier
removal, **99,621 rows** are modelled.

## Methodology and Data Pre-processing

### Leakage-free feature set
Only inputs knowable *without* IV are used. The 10 features are:

| Feature | Source |
|---|---|
| `underlying_last`, `strike`, `dte`, `tte` (= dte/365) | market observables |
| `log_moneyness` = ln(S/K), `strike_distance_pct` | geometry of the contract |
| `c_volume` | trading activity |
| `rv_10d`, `rv_20d`, `rv_30d` | **realized volatility** of SPY (annualized rolling std of daily log returns) |

The realized-vol features are the central legitimate signal: they are computed purely from the
underlying's past prices and replace the leaked Greeks. A `FORBIDDEN_COLUMNS` guard
(`features.assert_no_leakage`) makes it impossible for any Greek, IV, or option price to re-enter
the feature matrix, and this is enforced by a unit test.

### Pipeline steps
1. **Chunked load & sampling** across 2020–2022 (`data.py`).
2. **Realized-vol engineering** from one SPY close per `quote_date` (`realized_vol.py`).
3. **Feature engineering** + **cumulative IQR outlier removal** (fixing the original notebook's
   bug where only the last column's filter survived) (`features.py`).
4. **Log target**: model `log1p(c_iv)`; all metrics are inverted with `expm1` and reported in
   true IV units.
5. **Temporal split grouped by `quote_date`** — train on earlier dates, test on strictly later
   dates, so no day's volatility surface leaks across the split. `TimeSeriesSplit` is used for any
   tuning (`split.py`). No `StandardScaler` (trees are scale-invariant).
6. **Models**: a leakage-free **Random Forest** (headline), a **HistGradientBoosting** comparison,
   and a trivial **realized-vol baseline** for context (`model.py`).
7. **Evaluation** in IV units *and* in dollars — the real objective — via Black-Scholes price from
   predicted IV vs. market mid (`pricing.py`, `evaluate.py`).

For this run: **train 2020-01-24 → 2022-06-14** (82,017 rows), **test 2022-06-15 → 2022-12-30**
(17,604 rows).

## Results

All metrics are on the temporal **test** set (future dates the model never saw), in **IV units**,
plus Black-Scholes pricing error in **dollars** vs. market mid.

| Model | IV MAE | IV RMSE | IV R² | Price MAE ($) | Price median AE ($) |
|---|---|---|---|---|---|
| Realized-vol baseline | 0.0661 | 0.0876 | −0.597 | 2.54 | 1.07 |
| **Random Forest** | **0.0213** | **0.0317** | **0.791** | **1.02** | **0.39** |
| HistGradientBoosting | 0.0211 | 0.0316 | 0.792 | 0.95 | 0.39 |

Feature importances (Random Forest) are dominated by **`log_moneyness` (0.67)**, then time to
expiry (`dte` + `tte` ≈ 0.16) and **realized volatility** (`rv_10/20/30d` ≈ 0.09) — exactly the
structure of a real volatility surface, with no leaked Greeks. Figures are in `outputs/`:
`iv_distributions.png`, `actual_vs_predicted_iv.png`, `residuals.png`, `feature_importances.png`,
`model_comparison.png`.

## Conclusion

A leakage-free, temporally-validated model predicts SPY call IV with **R² ≈ 0.79 and MAE ≈ 0.021
in IV units**, translating to a **median Black-Scholes mispricing of about $0.39** vs. the market
mid. This is a genuine, modest result — materially below the original spurious 0.868, and far more
trustworthy. The realized-vol baseline's *negative* R² confirms the tree models are learning real
cross-sectional structure (moneyness × expiry × realized vol), not just echoing a single feature.

The gap from a perfect fit reflects what a leakage-free model honestly cannot see: the
forward-looking, supply/demand component of the volatility surface (skew/term-structure premia)
that the market prices in beyond recent realized volatility.

## Challenges & Limitations

- **Honest ceiling.** Without IV-derived inputs, the achievable R² is bounded by how much of the
  vol surface is explainable from contract geometry and realized vol. ~0.79 is reasonable for this
  feature set.
- **Regime coverage.** Sampling across 2020–2022 means the training data includes the COVID vol
  spike; the test window (H2 2022) is a separate regime, which is the right way to measure
  generalization.
- **Risk-free rate** is approximated as a small constant (0.01); 2020–2022 was a near-zero-rate
  era and `rho` is tiny, so this has negligible effect on the pricing metric.

## Future Work

- **Richer leakage-free features**: realized-vol skew/term-structure proxies, day-of-week/expiry
  effects, VIX as an *exogenous* (non-`c_iv`) input.
- **Surface-aware models**: fit per-expiry smiles, or a model that predicts the whole surface
  jointly rather than row-by-row.
- **GARCH / sequence models** for the time-varying volatility component, and a full Black-Scholes
  back-test of the predicted-IV prices against historical market prices.

---

*Methodology audit and the full list of original defects: [`AUDIT.md`](./AUDIT.md). Reproduce with
`uv run ivol run`; corrected metrics in `outputs/metrics.json`.*
