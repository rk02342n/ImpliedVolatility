"""Leakage-free feature engineering and the central leakage guard.

The guard (`assert_no_leakage`) is the single most important audit fix: it makes it
impossible to silently reintroduce a Greek or price column into the feature matrix
(AUDIT.md §1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FORBIDDEN_COLUMNS, LOG_TARGET, TARGET, Config


class LeakageError(AssertionError):
    """Raised when an IV-derived column would enter the feature matrix."""


def assert_no_leakage(feature_columns: list[str]) -> None:
    """Fail loudly if any forbidden (IV-derived) column is used as a feature."""
    leaked = sorted(set(feature_columns) & FORBIDDEN_COLUMNS)
    if leaked:
        raise LeakageError(
            f"Forbidden IV-derived columns found in features: {leaked}. "
            "These are computed from c_iv and would leak the target (see AUDIT.md §1)."
        )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived, leakage-free columns and the log target."""
    df = df.copy()
    df["tte"] = df["dte"] / 365.0
    df["log_moneyness"] = np.log(df["underlying_last"] / df["strike"])
    df[LOG_TARGET] = np.log1p(df[TARGET])
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame:
    """Cumulative IQR outlier filter.

    Fixes the original notebook's bug where each iteration re-sliced the *original*
    frame, so only the last column's filter survived (AUDIT.md §3).
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask &= df[col].between(lower, upper)
    return df[mask]


def build_xy(
    df: pd.DataFrame, config: Config, drop_outliers: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y_log, meta) ready for modelling.

    ``meta`` carries non-feature columns needed downstream (quote_date for the temporal
    split; c_bid/c_ask/strike/underlying/tte/c_iv for the dollar pricing metric).
    """
    df = engineer_features(df)
    feature_columns = config.feature_columns
    assert_no_leakage(feature_columns)

    needed = feature_columns + [LOG_TARGET, TARGET, "quote_date", "c_bid", "c_ask"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    if drop_outliers:
        df = remove_outliers_iqr(df, feature_columns + [LOG_TARGET])

    df = df.reset_index(drop=True)
    X = df[feature_columns].copy()
    assert_no_leakage(list(X.columns))
    y_log = df[LOG_TARGET].copy()

    meta_cols = ["quote_date", TARGET, "c_bid", "c_ask", "strike", "underlying_last", "tte"]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()
    return X, y_log, meta
