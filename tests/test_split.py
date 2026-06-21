"""Temporal-split correctness tests (AUDIT.md §2a)."""

import numpy as np
import pandas as pd

from implied_volatility.split import temporal_split


def _make():
    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        np.repeat(np.arange(50), 20), unit="D"
    )
    n = len(dates)
    X = pd.DataFrame({"f": np.arange(n, dtype=float)})
    y = pd.Series(np.arange(n, dtype=float))
    return X, y, pd.Series(dates)


def test_train_dates_strictly_precede_test_dates():
    X, y, dates = _make()
    _, _, _, _, train_idx, test_idx = temporal_split(X, y, dates, test_fraction=0.2)
    train_dates = pd.to_datetime(dates.loc[train_idx])
    test_dates = pd.to_datetime(dates.loc[test_idx])
    assert train_dates.max() < test_dates.min()


def test_no_date_straddles_the_split():
    X, y, dates = _make()
    _, _, _, _, train_idx, test_idx = temporal_split(X, y, dates, test_fraction=0.3)
    train_set = set(pd.to_datetime(dates.loc[train_idx]).unique())
    test_set = set(pd.to_datetime(dates.loc[test_idx]).unique())
    assert train_set & test_set == set()


def test_test_fraction_is_respected_by_dates():
    X, y, dates = _make()
    _, _, _, _, train_idx, test_idx = temporal_split(X, y, dates, test_fraction=0.2)
    n_test_dates = pd.to_datetime(dates.loc[test_idx]).nunique()
    assert n_test_dates == 10  # 20% of 50 distinct dates
