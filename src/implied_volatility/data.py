"""Loading and sampling the raw OptionsDX CSV.

The file is 1.28 GB / ~3.6M rows and is **not sorted by date** (rows are grouped by
expiry), so we read it in chunks and draw a seeded sample spread across the whole file —
fixing the original notebook's `nrows=40000` slice that only covered Aug-Sep 2021
(AUDIT.md §3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import USECOLS, Config


def normalize_columns(columns: pd.Index) -> dict[str, str]:
    """Map raw bracketed headers (e.g. ``[C_IV]``) to clean snake-ish names (``c_iv``)."""
    return {
        col: col.lower().replace("[", "").replace("]", "").replace(" ", "")
        for col in columns
    }


def _coerce_numeric(df: pd.DataFrame, exclude: tuple[str, ...] = ("quote_date",)) -> pd.DataFrame:
    """Coerce object columns to numeric, turning blanks into NaN."""
    for col in df.columns:
        if col in exclude:
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    return df


def _read_header(csv_path: Path) -> dict[str, str]:
    header = pd.read_csv(csv_path, nrows=0)
    return normalize_columns(header.columns)


def load_sample(config: Config, csv_path: Path | None = None) -> pd.DataFrame:
    """Read the CSV in chunks and return a date-spread sample of valid call rows.

    Valid = non-null ``c_iv`` and ``dte > 0`` (expiry-day rows have ``tte=0`` which
    breaks Black-Scholes and the realized-vol join).
    """
    csv_path = Path(csv_path) if csv_path is not None else config.raw_csv
    rename = _read_header(csv_path)
    # Map our desired clean names back to the raw header text to pass to usecols.
    raw_for_clean = {clean: raw for raw, clean in rename.items()}
    usecols_raw = [raw_for_clean[c] for c in USECOLS if c in raw_for_clean]

    rng = np.random.default_rng(config.random_state)
    kept: list[pd.DataFrame] = []
    total_valid = 0

    reader = pd.read_csv(
        csv_path,
        engine="c",
        usecols=usecols_raw,
        chunksize=config.chunksize,
        dtype=str,  # read raw; coerce ourselves (avoids mixed-dtype inference across chunks)
    )
    for chunk in reader:
        chunk = chunk.rename(columns=normalize_columns(chunk.columns))
        chunk = _coerce_numeric(chunk)
        chunk["quote_date"] = pd.to_datetime(chunk["quote_date"], errors="coerce")
        valid = chunk[
            chunk["c_iv"].notna()
            & (chunk["c_iv"] > 0)
            & chunk["dte"].notna()
            & (chunk["dte"] > 0)
            & chunk["quote_date"].notna()
        ]
        if valid.empty:
            continue
        total_valid += len(valid)
        kept.append(valid)

    df = pd.concat(kept, ignore_index=True)
    if len(df) > config.sample_size:
        idx = rng.choice(len(df), size=config.sample_size, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
    return df


def load_underlying_series(config: Config, csv_path: Path | None = None) -> pd.Series:
    """One SPY close per ``quote_date`` across the entire file (cheap two-column pass).

    Used to engineer realized volatility independently of the modelling sample.
    """
    csv_path = Path(csv_path) if csv_path is not None else config.raw_csv
    rename = _read_header(csv_path)
    raw_for_clean = {clean: raw for raw, clean in rename.items()}
    usecols_raw = [raw_for_clean["quote_date"], raw_for_clean["underlying_last"]]

    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        csv_path,
        engine="c",
        usecols=usecols_raw,
        chunksize=config.chunksize,
        dtype=str,
    )
    for chunk in reader:
        chunk = chunk.rename(columns=normalize_columns(chunk.columns))
        chunk["quote_date"] = pd.to_datetime(chunk["quote_date"], errors="coerce")
        chunk["underlying_last"] = pd.to_numeric(chunk["underlying_last"], errors="coerce")
        parts.append(chunk.dropna(subset=["quote_date", "underlying_last"]))

    allrows = pd.concat(parts, ignore_index=True)
    series = (
        allrows.groupby("quote_date")["underlying_last"]
        .first()
        .sort_index()
    )
    series.name = "underlying_last"
    return series
