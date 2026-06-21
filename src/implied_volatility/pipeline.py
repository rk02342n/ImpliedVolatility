"""End-to-end leakage-free pipeline: load → features → temporal split → train → evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOG_TARGET, Config
from .data import load_sample, load_underlying_series
from .evaluate import (
    iv_metrics,
    save_actual_vs_predicted,
    save_feature_importances,
    save_iv_distributions,
    save_model_comparison,
    save_residuals,
)
from .features import build_xy
from .model import (
    RealizedVolBaseline,
    build_hist_gbr,
    build_random_forest,
)
from .pricing import pricing_error
from .realized_vol import attach_realized_vol
from .split import temporal_split


def _sort_by_date(*frames, dates: pd.Series):
    """Return frames reindexed into chronological order (stable)."""
    order = np.argsort(dates.to_numpy(), kind="stable")
    return [f.iloc[order].reset_index(drop=True) for f in frames]


def run(config: Config, csv_path: Path | None = None, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    log("[1/6] Loading underlying price series (full-file pass)…")
    underlying = load_underlying_series(config, csv_path)
    log(f"      {len(underlying)} trading days, "
        f"{underlying.index.min().date()} → {underlying.index.max().date()}")

    log("[2/6] Sampling option rows across the full date range…")
    df = load_sample(config, csv_path)
    log(f"      {len(df):,} valid call rows sampled")

    log("[3/6] Engineering realized-vol + leakage-free features…")
    df = attach_realized_vol(df, underlying, config)
    X, y_log, meta = build_xy(df, config)
    log(f"      Features ({len(X.columns)}): {list(X.columns)}")
    log("      Leakage guard passed: no Greek/price column present.")

    log("[4/6] Temporal split grouped by quote_date…")
    # Sort everything chronologically so TimeSeriesSplit folds also respect time.
    X, y_log, meta = _sort_by_date(X, y_log.to_frame(), meta, dates=meta["quote_date"])
    y_log = y_log[LOG_TARGET]
    X_train, X_test, y_train, y_test, train_idx, test_idx = temporal_split(
        X, y_log, meta["quote_date"], test_fraction=config.test_fraction
    )
    meta_test = meta.loc[test_idx]
    log(f"      Train: {meta.loc[train_idx, 'quote_date'].min().date()} → "
        f"{meta.loc[train_idx, 'quote_date'].max().date()} ({len(X_train):,} rows)")
    log(f"      Test:  {meta_test['quote_date'].min().date()} → "
        f"{meta_test['quote_date'].max().date()} ({len(X_test):,} rows)")

    models = {
        "realized_vol_baseline": RealizedVolBaseline(rv_column=config.realized_vol_columns[0]),
        "random_forest": build_random_forest(config.random_state),
        "hist_gradient_boosting": build_hist_gbr(config.random_state),
    }

    log("[5/6] Training & evaluating models (IV units + $ pricing error)…")
    metrics_by_model: dict[str, dict] = {}
    rf_predictions = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        m = iv_metrics(y_test.to_numpy(), y_pred_log)

        iv_pred = np.expm1(y_pred_log)
        price = pricing_error(
            S=meta_test["underlying_last"].to_numpy(),
            K=meta_test["strike"].to_numpy(),
            t=meta_test["tte"].to_numpy(),
            predicted_iv=iv_pred,
            c_bid=meta_test["c_bid"].to_numpy(),
            c_ask=meta_test["c_ask"].to_numpy(),
            r=config.risk_free_rate,
        )
        m.update(price)
        metrics_by_model[name] = m
        log(f"      {name:24s} IV R²={m['iv_r2']:.4f}  IV MAE={m['iv_mae']:.4f}  "
            f"price MAE=${m['price_mae_usd']:.3f}")
        if name == "random_forest":
            rf_predictions = (y_test.to_numpy(), y_pred_log, model)

    log("[6/6] Writing outputs (figures + metrics.json)…")
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    iv_all = np.expm1(y_log)
    save_iv_distributions(pd.Series(iv_all), pd.Series(y_log), out_dir)
    save_model_comparison(metrics_by_model, out_dir)

    if rf_predictions is not None:
        y_true_log, y_pred_log, rf_model = rf_predictions
        iv_true, iv_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
        save_actual_vs_predicted(iv_true, iv_pred, out_dir, "Random Forest")
        save_residuals(iv_true, iv_pred, out_dir)
        importances = dict(zip(X.columns, rf_model.feature_importances_, strict=True))
        save_feature_importances(importances, out_dir)
    else:
        importances = {}

    result = {
        "config": {
            "sample_size": config.sample_size,
            "n_rows_used": int(len(X)),
            "test_fraction": config.test_fraction,
            "risk_free_rate": config.risk_free_rate,
            "features": list(X.columns),
            "train_date_range": [
                str(meta.loc[train_idx, "quote_date"].min().date()),
                str(meta.loc[train_idx, "quote_date"].max().date()),
            ],
            "test_date_range": [
                str(meta_test["quote_date"].min().date()),
                str(meta_test["quote_date"].max().date()),
            ],
        },
        "metrics": metrics_by_model,
        "random_forest_feature_importances": {
            k: float(v) for k, v in sorted(importances.items(), key=lambda kv: -kv[1])
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log(f"      Wrote {out_dir / 'metrics.json'} and figures.")
    return result
