"""Realized volatility of the underlying — the core *legitimate* predictor of IV.

Unlike the Greeks (which are derived from IV), realized vol is computed only from the
underlying's past prices and is therefore knowable at prediction time. It is the honest
signal that replaces the leaked Greeks (AUDIT.md §1, §4 / checklist #1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR, Config


def compute_realized_vol(underlying: pd.Series, windows: tuple[int, ...]) -> pd.DataFrame:
    """Annualized rolling realized volatility of daily log returns.

    Parameters
    ----------
    underlying : Series indexed by ``quote_date`` (sorted) of the underlying close.
    windows : rolling windows in trading days.

    Returns a DataFrame indexed by date with one ``rv_{w}d`` column per window.
    """
    underlying = underlying.sort_index()
    log_ret = np.log(underlying / underlying.shift(1))
    out = pd.DataFrame(index=underlying.index)
    for w in windows:
        out[f"rv_{w}d"] = log_ret.rolling(window=w, min_periods=max(2, w // 2)).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    return out


def attach_realized_vol(
    df: pd.DataFrame, underlying: pd.Series, config: Config
) -> pd.DataFrame:
    """Left-join realized-vol features onto option rows by ``quote_date``."""
    rv = compute_realized_vol(underlying, config.realized_vol_windows)
    merged = df.merge(rv, left_on="quote_date", right_index=True, how="left")
    return merged
