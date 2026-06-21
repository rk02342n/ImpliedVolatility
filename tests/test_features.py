"""Leakage-guard and feature-engineering tests (the central audit fix)."""

import numpy as np
import pandas as pd
import pytest

from implied_volatility.config import FORBIDDEN_COLUMNS, Config
from implied_volatility.features import (
    LeakageError,
    assert_no_leakage,
    build_xy,
    remove_outliers_iqr,
)
from implied_volatility.realized_vol import attach_realized_vol


def _toy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2021-01-04") + pd.to_timedelta(
        rng.integers(0, 40, size=n), unit="D"
    )
    return pd.DataFrame(
        {
            "quote_date": dates,
            "underlying_last": rng.uniform(380, 460, n),
            "strike": rng.uniform(350, 480, n),
            "dte": rng.uniform(1, 200, n),
            "c_iv": rng.uniform(0.1, 0.6, n),
            "c_volume": rng.integers(0, 5000, n).astype(float),
            "c_bid": rng.uniform(0.5, 30, n),
            "c_ask": rng.uniform(0.5, 30, n),
            "strike_distance_pct": rng.uniform(0, 0.3, n),
            # Deliberately include Greeks to prove they are excluded.
            "c_delta": rng.uniform(0, 1, n),
            "c_vega": rng.uniform(0, 1, n),
        }
    )


def _toy_underlying():
    idx = pd.date_range("2020-12-01", "2021-02-28", freq="B")
    prices = 400 + np.cumsum(np.random.default_rng(1).normal(0, 2, len(idx)))
    return pd.Series(prices, index=idx, name="underlying_last")


def test_assert_no_leakage_rejects_greeks():
    with pytest.raises(LeakageError):
        assert_no_leakage(["log_moneyness", "c_vega"])


def test_build_xy_excludes_all_forbidden_columns():
    cfg = Config(sample_size=1000)
    df = attach_realized_vol(_toy_df(), _toy_underlying(), cfg)
    X, y_log, meta = build_xy(df, cfg)
    assert set(X.columns) & FORBIDDEN_COLUMNS == set()
    assert "c_iv" not in X.columns
    assert all(g not in X.columns for g in ("c_delta", "c_vega"))
    # Realized-vol features are present as legitimate predictors.
    assert all(c in X.columns for c in cfg.realized_vol_columns)
    assert len(X) == len(y_log) == len(meta)


def test_realized_vol_join_aligns_by_date():
    cfg = Config()
    df = _toy_df()
    underlying = _toy_underlying()
    merged = attach_realized_vol(df, underlying, cfg)
    # Each row's rv should equal the realized-vol value for its own quote_date.
    from implied_volatility.realized_vol import compute_realized_vol

    rv = compute_realized_vol(underlying, cfg.realized_vol_windows)
    # Check every row's rv matches the value for its own date (NaN if date absent).
    expected = rv["rv_10d"].reindex(merged["quote_date"]).to_numpy()
    actual = merged["rv_10d"].to_numpy()
    assert np.allclose(actual, expected, equal_nan=True)


def test_remove_outliers_iqr_is_cumulative():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 1000], "b": [1, 2, 3, 4, 5]})
    out = remove_outliers_iqr(df, ["a", "b"])
    # The 1000 outlier in 'a' must be removed (the old bug kept only the last column).
    assert 1000 not in out["a"].to_numpy()
