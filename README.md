# Implied Volatility Prediction (leakage-free rebuild)

Predict SPY call-option **implied volatility (IV)** from inputs that are actually
knowable *before* you observe IV, then price options with Black-Scholes using the
predicted IV.

This is a rebuild of an earlier notebook whose headline result (R² = 0.868) was
**spurious target leakage**: it predicted IV from the option Greeks, but in this
OptionsDX dataset every Greek is computed *from* IV via Black-Scholes. See
[`AUDIT.md`](./AUDIT.md) for the full diagnosis and [`REPORT.md`](./REPORT.md) for the
corrected results.

## What changed vs. the original notebook

| Audit issue | Fix in this project |
|---|---|
| Greeks leak the target (§1) | Greeks + prices **dropped**; a `FORBIDDEN_COLUMNS` guard asserts they never enter `X` |
| Shuffled split on grouped time series (§2a) | **Temporal split** grouped by `quote_date`; `TimeSeriesSplit` for tuning |
| Scaler refit on test (§2b) | No scaler (trees are scale-invariant) |
| Grid search on 1,000 rows (§2c) | Time-aware tuning on a representative sample |
| Narrow first-40k-rows slice (§3) | **Chunked sample across all of 2020–2022** |
| Broken outlier filter (§3) | Cumulative IQR filtering |
| Metrics in log space (§3) | Metrics reported in **IV units** + **Black-Scholes price error in $** |
| No real predictive signal | Engineered **realized volatility** of the underlying |

## Setup

```bash
uv sync --extra dev
```

The raw dataset `spy_2020_2022.csv` (1.28 GB) is git-ignored; place it in the project
root.

## Run the pipeline

```bash
uv run ivol run                       # default: 200k rows sampled across 2020-2022
uv run ivol run --sample-size 50000   # faster iteration
uv run ivol run --csv /path/to/spy_2020_2022.csv
```

Outputs land in `outputs/`: `metrics.json` and figures (IV distributions, actual vs.
predicted, residuals, feature importances).

## Tests

```bash
uv run pytest
```

## Package layout

```
src/implied_volatility/
  config.py        # paths, leakage guard (FORBIDDEN_COLUMNS), feature list, hyper-params
  data.py          # chunked load, column normalization, call-only filter, date-spread sampling
  realized_vol.py  # SPY date->price series; rolling annualized realized-vol features
  features.py      # leakage-free feature engineering + outlier removal + guard
  split.py         # temporal split grouped by quote_date; TimeSeriesSplit factory
  pricing.py       # vectorized Black-Scholes call price; dollar price-error metric
  model.py         # RandomForest / HistGradientBoosting / realized-vol baseline
  evaluate.py      # IV-unit + dollar metrics; figures
  pipeline.py      # end-to-end orchestration
  cli.py           # `ivol run`
```
