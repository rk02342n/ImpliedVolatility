"""Black-Scholes pricing and the dollar-denominated objective metric (AUDIT.md §4).

The project's real goal is *pricing*: predict IV, plug it into Black-Scholes, and compare
the resulting call price to the market mid. Measuring error in dollars (not log-IV space)
is the metric that reflects what we actually care about.

``scipy`` ships transitively with scikit-learn, so ``ndtr`` (the normal CDF) is available
without adding a dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def black_scholes_call(
    S: np.ndarray,
    K: np.ndarray,
    t: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.01,
) -> np.ndarray:
    """Vectorized Black-Scholes call price.

    C = N(d1)·S − N(d2)·K·e^{−rt}, with d1 = (ln(S/K) + (r + σ²/2)t) / (σ√t).

    Guards against σ√t = 0 (expiry / zero-vol) by falling back to the discounted
    intrinsic value.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    vol_t = sigma * np.sqrt(t)
    intrinsic = np.maximum(S - K * np.exp(-r * t), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / vol_t
        d2 = d1 - vol_t
        price = ndtr(d1) * S - ndtr(d2) * K * np.exp(-r * t)

    return np.where(vol_t > 0, price, intrinsic)


def market_mid(c_bid: np.ndarray, c_ask: np.ndarray) -> np.ndarray:
    """Mid price; falls back to whichever side is present if one is zero/NaN."""
    c_bid = np.asarray(c_bid, dtype=float)
    c_ask = np.asarray(c_ask, dtype=float)
    mid = (c_bid + c_ask) / 2.0
    return mid


def pricing_error(
    S: np.ndarray,
    K: np.ndarray,
    t: np.ndarray,
    predicted_iv: np.ndarray,
    c_bid: np.ndarray,
    c_ask: np.ndarray,
    r: float = 0.01,
) -> dict[str, float]:
    """Dollar error of Black-Scholes price (using predicted IV) vs market mid.

    Returns MAE, median absolute error, and RMSE in dollars over rows with a valid
    (positive) market mid.
    """
    model_price = black_scholes_call(S, K, t, predicted_iv, r=r)
    mid = market_mid(c_bid, c_ask)
    valid = np.isfinite(mid) & (mid > 0) & np.isfinite(model_price)
    err = np.abs(model_price[valid] - mid[valid])
    return {
        "price_mae_usd": float(np.mean(err)),
        "price_median_ae_usd": float(np.median(err)),
        "price_rmse_usd": float(np.sqrt(np.mean(err**2))),
        "n_priced": int(valid.sum()),
    }
