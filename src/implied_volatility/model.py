"""Models: leakage-free Random Forest (headline), gradient boosting, and a baseline.

Tuning uses time-aware CV on a representative sample, replacing the original notebook's
grid search on the first 1,000 rows with shuffled folds (AUDIT.md §2c).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from .split import time_series_cv


class RealizedVolBaseline(BaseEstimator, RegressorMixin):
    """Trivial baseline: predict log1p(IV) from the shortest-window realized vol.

    Realized vol is itself an annualized volatility, so log1p(rv) is a sensible naive
    estimate of log1p(IV). This gives the tree models something honest to beat.
    """

    def __init__(self, rv_column: str = "rv_10d"):
        self.rv_column = rv_column

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless; store the column index for array fallbacks.
        self.rv_column_ = self.rv_column
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rv = X[self.rv_column_].to_numpy(dtype=float)
        return np.log1p(np.clip(rv, 0.0, None))


def build_random_forest(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )


def build_hist_gbr(random_state: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=None,
        learning_rate=0.1,
        max_iter=400,
        l2_regularization=1.0,
        random_state=random_state,
    )


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_splits: int = 4,
) -> GridSearchCV:
    """Light, time-aware grid search.

    Assumes ``X_train``/``y_train`` are already sorted by ``quote_date`` so the
    ``TimeSeriesSplit`` folds respect chronological order.
    """
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 16],
        "min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=random_state),
        param_grid=param_grid,
        cv=time_series_cv(n_splits=n_splits),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search
