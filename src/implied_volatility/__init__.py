"""Leakage-free implied-volatility prediction for SPY options.

See AUDIT.md for why the original notebook's R^2=0.868 was spurious (target leakage
through the Greeks). This package rebuilds the pipeline so IV is predicted only from
inputs knowable *without* IV, validated with a temporal split, and reported in true IV
units plus Black-Scholes price error in dollars.
"""

__version__ = "0.1.0"
