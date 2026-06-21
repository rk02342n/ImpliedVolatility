# ML Audit — Implied Volatility Prediction

**Reviewer:** Senior ML engineering review
**Artifacts reviewed:** `ImpliedVolatilityPrediction.ipynb`, `ImpliedVolatility_Report.pdf`, `spy_2020_2022.csv` (1.28 GB)
**Question asked:** *Is the logic for finding and predicting IV correct, or is it circular?*

## TL;DR

**The logic is circular.** The model predicts implied volatility (IV) from the option **Greeks**,
but in this dataset the Greeks are themselves computed from IV via Black-Scholes. The model is
therefore recovering the target from transformations of the target — textbook **target leakage** —
not learning anything predictive. The headline **R² = 0.868 is spurious**, and the pipeline cannot
serve its stated goal (predict IV → price options with Black-Scholes), because at pricing time you
would not yet know IV and so could not have the Greeks either.

This was verified both analytically and empirically against the provided dataset.

---

## 1. CRITICAL — Circular logic / target leakage

### Why it's circular
The feature set keeps the Greeks: `c_delta, c_gamma, c_vega, c_theta, c_rho`. These are **not
independent market observables**. In OptionsDX-style chains, the data vendor computes each Greek
from the Black-Scholes model using the implied volatility `c_iv` (σ) as an input. Every Greek is a
function of σ:

- `vega = S·φ(d₁)·√t`
- `gamma = φ(d₁) / (S·σ·√t)`
- `delta = N(d₁)`, where `d₁` itself contains σ
- `theta`, `rho` likewise depend on σ

There is even a closed-form way to invert IV out of two of the features:

```
vega / gamma = S² · σ · t      ⟹      σ = 100 · vega / (gamma · S² · t)
```

(The factor 100 is because vega is quoted per 1% vol move.)

Removing `c_last`, `c_bid`, `c_ask` (done in cell 5) was correct — those prices are what IV is
inverted from. But the Greeks are a **more direct leak** and were left in.

### Empirical proof (on `spy_2020_2022.csv`)
| Test | Result | Interpretation |
|---|---|---|
| Median of `c_iv ÷ [vega/(gamma·S²·t)]` | **99.8** | Confirms the closed-form identity (≈100) |
| RF using **Greeks only** → R²(log IV) | **0.986** | The 5 IV-derived features alone reconstruct the target |
| RF **with** Greeks (current model) | 0.993 | Greeks dominate; the "model" is an IV inverter |
| RF **leakage-free** (no Greeks), random split | 0.964 | Achievable without leakage (but see §2) |

The original notebook's own **feature-importance table** corroborates this: `strike (0.34)`,
`c_rho (0.13)`, `c_gamma (0.13)`, `c_vega (0.09)` — three of the top four features are Greeks.

### Consequence
1. The reported performance does not reflect any genuine predictive skill.
2. The model is **unusable for the stated goal**. To price an option you don't yet have a market
   price for, you don't know its IV — and therefore can't compute its Greeks. So the inputs the
   model relies on don't exist at prediction time.

### Fix
Drop **all** Greeks and **all** price columns. Predict IV only from inputs knowable without IV:
moneyness, strike, underlying price, time-to-expiry, plus engineered **realized volatility** of the
underlying and a risk-free rate. (See `quizzical-splashing-haven.md` plan, Phase 2.)

---

## 2. HIGH — Validation leakage (inflated metrics even without the Greeks)

### 2a. Random split on time-series, grouped data
`train_test_split(..., shuffle=True)` (cell 20) scatters rows from the **same `quote_date`, same
expiry, and adjacent strikes** across both train and test. The model memorizes each day's
volatility surface and is then tested on near-duplicate rows from that same surface.

**Empirical (leakage-free features, 200k rows):**
- Random split → R² = **0.969**
- Temporal split (train earlier dates, test later) → R² = **0.924**

The gap is the optimism injected by the random split. **Fix:** temporal split grouped by
`quote_date`; use `TimeSeriesSplit` for any tuning.

### 2b. Scaler refit on the test set
Cell 23: `X_test_scaled = scaler.fit_transform(X_test)` refits the scaler on test data; it should
be `scaler.transform(X_test)`. Harmless for a Random Forest (trees are scale-invariant — so the
StandardScaler is unnecessary here in the first place), but it is leakage and indicates a
methodology gap that would bite with scale-sensitive models. **Fix:** drop the scaler for RF, or
fit on train only.

### 2c. GridSearchCV tuned on 1,000 rows
Cell 30 tunes on `X_train_scaled[:1000]` then refits on the full set. The "best params" are derived
from a tiny, non-representative slice, and the CV ignores time ordering. The report notes that
tuning made metrics slightly *worse* — a classic overfit/leakage signal. **Fix:** tune on a
representative sample with time-aware CV.

---

## 3. MEDIUM — Data handling & reporting

- **Broken outlier function** (cell 12): `remove_outliers_iqr` reassigns `data = df[...]` from the
  *original* `df` on every loop iteration, so only the **last** column's filter is applied, not the
  intended cumulative filtering. It also references `X.columns`, but `X` is first defined in cell 20
  — so the notebook only runs if cells are executed out of order. Reproducibility hazard. **Fix:**
  filter cumulatively (`df = df[mask]` each iteration) and define column lists before use.
- **Metrics reported in log space.** MAE 0.0208 / MSE 0.00143 are on `log1p(IV)`, not IV. The
  report's claim that predictions are "close to actual market-implied volatilities" conflates the
  two scales. **Fix:** invert with `expm1` and report errors in IV units (and in price/$).
- **Narrow, mislabeled data slice.** `nrows=40000` with a comment claiming "data from 2020"; the
  actual first rows are **Aug–Sep 2021** (~39 trading days). The model never sees a volatility
  regime change (e.g., the 2020 COVID spike), so generalization is untested. **Fix:** sample across
  the full 2020–2022 range with a chunked read.
- **Unit inconsistency:** code computes `tte = dte/365` (years) while the report/comment says
  `dte*365`. The IV-vs-DTE scatterplot's x-axis is labeled "Days to Expiry" but plots standardized
  values (−1..3). **Fix:** standardize and document units.
- **Collinear derived features:** `moneyness`, `log_moneyness`, `strike_distance`,
  `strike_distance_pct` are all functions of S and K. Redundant (harmless for RF; prune for clarity).
- **Report/notebook mismatch:** the report cites `mutual_info_regression` as justification for the
  tree-based approach, but that code is not in the notebook.

---

## 4. Conceptual framing

The intended use is **option pricing**: predict IV → plug into Black-Scholes. That makes this a
**cross-sectional volatility-surface reconstruction** problem: given a strike and expiry at a point
in time, estimate IV from information available *without* IV, then price. The correct framing
forbids every IV-derived input and demands a leakage-free, time-respecting validation — plus a
pricing metric (predicted-IV Black-Scholes price vs market mid, in dollars) that measures the thing
you actually care about.

---

## Prioritized remediation checklist

1. **[Critical]** Remove all Greeks and price columns from features.
2. **[High]** Switch to a temporal split grouped by `quote_date`; time-aware CV.
3. **[High]** Sample across the full 2020–2022 date range, not the first 40k rows.
4. **[Medium]** Fix `remove_outliers_iqr`; remove the StandardScaler (or fit on train only).
5. **[Medium]** Report metrics in IV units and add a leakage-free baseline for context.
6. **[Medium]** Add the real objective metric: Black-Scholes price from predicted IV vs market mid ($).
7. **[Low]** Reconcile report vs notebook; fix unit labels; prune collinear features.
