"""Temporal, date-grouped train/test split (AUDIT.md §2a).

A shuffled split scatters rows from the same ``quote_date``/expiry/adjacent strikes
across train and test, so the model memorizes each day's volatility surface and is then
tested on near-duplicates of it. We instead hold out the *latest* distinct dates, so all
training dates strictly precede all test dates and no date straddles the boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index, pd.Index]:
    """Split by distinct ``quote_date`` so train dates < test dates.

    Returns ``(X_train, X_test, y_train, y_test, train_idx, test_idx)``.
    """
    dates = pd.to_datetime(dates)
    unique_dates = np.sort(dates.unique())
    n_test_dates = max(1, int(round(len(unique_dates) * test_fraction)))
    cutoff = unique_dates[-n_test_dates]  # first date that belongs to the test set

    test_mask = dates.to_numpy() >= cutoff
    train_idx = X.index[~test_mask]
    test_idx = X.index[test_mask]

    return (
        X.loc[train_idx],
        X.loc[test_idx],
        y.loc[train_idx],
        y.loc[test_idx],
        train_idx,
        test_idx,
    )


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Time-aware CV splitter for tuning (replaces shuffled k-fold; AUDIT.md §2c).

    Note: rows must be pre-sorted by ``quote_date`` for the folds to respect time.
    """
    return TimeSeriesSplit(n_splits=n_splits)
