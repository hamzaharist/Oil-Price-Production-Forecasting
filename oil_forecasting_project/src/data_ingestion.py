# Data ingestion — fetch Brent crude prices and U.S. production data

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# --- Brent crude prices (Yahoo Finance, ticker BZ=F) ---

def fetch_brent_prices(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    # Download daily Brent futures and resample to monthly average
    if cache_path and Path(cache_path).exists():
        daily = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not isinstance(daily.index, pd.DatetimeIndex):
            daily.index = pd.to_datetime(daily.index)
    else:
        try:
            import yfinance as yf
            ticker = yf.Ticker("BZ=F")
            daily = ticker.history(start=start, end=end)[["Close"]]
            daily.columns = ["price_brent"]
            # strip timezone info
            if hasattr(daily.index, 'tz') and daily.index.tz is not None:
                daily.index = daily.index.tz_localize(None)
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                daily.to_csv(cache_path)
        except Exception as exc:
            warnings.warn(f"yfinance failed ({exc}); using synthetic data.")
            daily = _synthetic_brent(start, end)

    # ensure tz-naive
    if hasattr(daily.index, 'tz') and daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)

    monthly = daily.resample("MS").mean().rename(
        columns=lambda c: "price_brent" if c != "price_brent" else c
    )
    monthly.index.name = "date"
    monthly = monthly[["price_brent"]].dropna()
    return monthly


# --- EIA U.S. crude-oil production (thousand bbl/day) ---

def fetch_eia_production(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    # Return monthly U.S. production; generates synthetic if no cache
    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    df = _synthetic_production(start, end)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
    return df


# --- Combined dataset ---

def build_dataset(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    # Merge price and production into one monthly DataFrame
    cache_dir = Path(cache_dir) if cache_dir else None
    prices = fetch_brent_prices(
        start, end,
        cache_path=cache_dir / "brent_daily.csv" if cache_dir else None,
    )
    production = fetch_eia_production(
        start, end,
        cache_path=cache_dir / "eia_production.csv" if cache_dir else None,
    )
    df = prices.join(production, how="inner")
    return df


# --- Synthetic data generators (deterministic seed) ---

def _synthetic_brent(start: str, end: str) -> pd.DataFrame:
    # Geometric Brownian motion calibrated to ~35% annual volatility
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    mu = 0.03 / 252
    sigma = 0.35 / np.sqrt(252)
    log_returns = mu + sigma * rng.standard_normal(n)
    log_returns[0] = 0.0
    prices = 28.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"price_brent": prices}, index=dates)


def _synthetic_production(start: str, end: str) -> pd.DataFrame:
    # Realistic U.S. production curve with shale boom + COVID dip
    rng = np.random.default_rng(123)
    dates = pd.date_range(start, end, freq="MS")
    n = len(dates)
    t = np.arange(n)

    base = 5800 + 25 * t

    # shale ramp (2010+)
    shale_start = max(0, (pd.Timestamp("2010-01-01") - pd.Timestamp(start)).days // 30)
    shale = np.zeros(n)
    if shale_start < n:
        shale[shale_start:] = 45 * np.arange(n - shale_start)

    trend = np.minimum(base + shale, 13_500)

    # COVID dip (2020-03 to 2020-09)
    covid_start = max(0, (pd.Timestamp("2020-03-01") - pd.Timestamp(start)).days // 30)
    covid_end = min(n, covid_start + 7)
    if covid_start < n:
        trend[covid_start:covid_end] *= np.linspace(0.82, 0.92, covid_end - covid_start)

    noise = rng.normal(0, 120, size=n)
    production = np.maximum(trend + noise, 4000)
    df = pd.DataFrame({"production_kbpd": production}, index=dates)
    df.index.name = "date"
    return df
