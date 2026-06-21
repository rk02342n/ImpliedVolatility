"""Black-Scholes sanity checks."""

import numpy as np

from implied_volatility.pricing import black_scholes_call, pricing_error


def test_call_at_least_intrinsic():
    S = np.array([100.0, 100.0])
    K = np.array([90.0, 110.0])
    t = np.array([0.5, 0.5])
    sigma = np.array([0.2, 0.2])
    price = black_scholes_call(S, K, t, sigma, r=0.01)
    intrinsic = np.maximum(S - K * np.exp(-0.01 * t), 0.0)
    assert np.all(price >= intrinsic - 1e-9)


def test_deep_itm_approaches_discounted_forward():
    # Very deep in the money, low vol: price ≈ S − K·e^{−rt}.
    S = np.array([1000.0])
    K = np.array([10.0])
    t = np.array([1.0])
    sigma = np.array([0.05])
    price = black_scholes_call(S, K, t, sigma, r=0.02)
    expected = S - K * np.exp(-0.02 * t)
    assert np.isclose(price[0], expected[0], rtol=1e-4)


def test_known_textbook_value():
    # S=100, K=100, t=1, sigma=0.2, r=0.05 -> ~10.4506 (standard reference value).
    price = black_scholes_call(
        np.array([100.0]), np.array([100.0]), np.array([1.0]), np.array([0.2]), r=0.05
    )
    assert np.isclose(price[0], 10.4506, atol=1e-3)


def test_zero_vol_falls_back_to_intrinsic():
    S = np.array([100.0])
    K = np.array([90.0])
    t = np.array([0.5])
    price = black_scholes_call(S, K, t, np.array([0.0]), r=0.01)
    expected = np.maximum(S - K * np.exp(-0.01 * t), 0.0)
    assert np.isclose(price[0], expected[0])


def test_vega_positive_via_finite_difference():
    args = (np.array([100.0]), np.array([100.0]), np.array([1.0]))
    p_lo = black_scholes_call(*args, np.array([0.20]), r=0.01)
    p_hi = black_scholes_call(*args, np.array([0.21]), r=0.01)
    assert p_hi[0] > p_lo[0]  # price increases with volatility


def test_pricing_error_reports_dollars():
    S = np.array([100.0, 100.0])
    K = np.array([100.0, 105.0])
    t = np.array([1.0, 1.0])
    iv = np.array([0.2, 0.2])
    mid = black_scholes_call(S, K, t, iv, r=0.01)
    out = pricing_error(S, K, t, iv, c_bid=mid - 0.05, c_ask=mid + 0.05, r=0.01)
    assert out["n_priced"] == 2
    assert out["price_mae_usd"] < 0.1  # perfect IV -> tiny error vs mid
