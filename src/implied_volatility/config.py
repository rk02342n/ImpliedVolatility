"""Central configuration: paths, columns, and hyper-parameters.

Keeping the leakage rules in one place (FORBIDDEN_COLUMNS / FEATURE_COLUMNS) makes the
single most important audit fix explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "spy_2020_2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- Raw column names (after normalization: lower-case, brackets/spaces stripped) ---
TARGET = "c_iv"
LOG_TARGET = "log_c_iv"
DATE_COL = "quote_date"
UNDERLYING_COL = "underlying_last"

# Columns we actually read from the 1.28 GB file. bid/ask are kept ONLY for the
# dollar-denominated Black-Scholes pricing metric — never used as model features.
USECOLS = [
    "quote_date",
    "underlying_last",
    "dte",
    "strike",
    "c_iv",
    "c_volume",
    "c_bid",
    "c_ask",
    "strike_distance_pct",
]

# --- Leakage guard (AUDIT.md §1) ---------------------------------------------
# Every Greek in this dataset is computed FROM c_iv via Black-Scholes, and the option
# prices are what c_iv is inverted from. None of these may ever enter the feature matrix.
FORBIDDEN_COLUMNS = frozenset(
    {
        "c_delta",
        "c_gamma",
        "c_vega",
        "c_theta",
        "c_rho",
        "c_iv",
        "log_c_iv",
        "c_last",
        "c_bid",
        "c_ask",
        # put-side analogues, for completeness
        "p_delta",
        "p_gamma",
        "p_vega",
        "p_theta",
        "p_rho",
        "p_iv",
        "p_last",
        "p_bid",
        "p_ask",
    }
)

# Leakage-free features knowable without IV. Realized-vol columns are appended at runtime.
BASE_FEATURE_COLUMNS = [
    "underlying_last",
    "strike",
    "dte",
    "tte",
    "log_moneyness",
    "strike_distance_pct",
    "c_volume",
]

# Rolling windows (trading days) for realized volatility of the underlying.
REALIZED_VOL_WINDOWS = (10, 20, 30)
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Config:
    """Run-time configuration knobs."""

    raw_csv: Path = RAW_CSV
    output_dir: Path = OUTPUT_DIR
    sample_size: int = 200_000
    chunksize: int = 250_000
    test_fraction: float = 0.2  # fraction of distinct dates held out as the temporal test set
    risk_free_rate: float = 0.01  # 2020-2022 was near zero-rate; rho is tiny anyway
    random_state: int = 42
    realized_vol_windows: tuple[int, ...] = field(default=REALIZED_VOL_WINDOWS)

    @property
    def realized_vol_columns(self) -> list[str]:
        return [f"rv_{w}d" for w in self.realized_vol_windows]

    @property
    def feature_columns(self) -> list[str]:
        return BASE_FEATURE_COLUMNS + self.realized_vol_columns
