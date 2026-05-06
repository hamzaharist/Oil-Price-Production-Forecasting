# Data ingestion — fetch Brent crude prices and macroeconomic features from EIA, FRED, and World Bank

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Brent crude prices (Yahoo Finance, ticker BZ=F)
# ---------------------------------------------------------------------------

def fetch_brent_prices(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Download daily Brent futures and resample to monthly average."""
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
            if hasattr(daily.index, "tz") and daily.index.tz is not None:
                daily.index = daily.index.tz_localize(None)
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                daily.to_csv(cache_path)
        except Exception as exc:
            warnings.warn(f"yfinance failed ({exc}); using synthetic data.")
            daily = _synthetic_brent(start, end)

    if hasattr(daily.index, "tz") and daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)

    monthly = daily.resample("MS").mean().rename(
        columns=lambda c: "price_brent" if c != "price_brent" else c
    )
    monthly.index.name = "date"
    return monthly[["price_brent"]].dropna()


# ---------------------------------------------------------------------------
# EIA — physical supply signals
# ---------------------------------------------------------------------------

# Series IDs for EIA API v2
_EIA_SERIES = {
    "production_kbpd":       "PET.MCRFPUS2.M",   # U.S. crude oil field production
    "inventory_commercial":  "PET.MCESTUS1.M",   # Commercial crude inventories (Mbbl)
    "inventory_spr":         "PET.MSSACUS1.M",   # SPR stock (Mbbl)
    "imports_kbpd":          "PET.MCRIMUS2.M",   # U.S. crude imports
    "exports_kbpd":          "PET.MCREXUS2.M",   # U.S. crude exports
    "refinery_util_pct":     "PET.MOPUEUS2.M",   # Refinery utilisation rate
    "product_supplied_kbpd": "PET.MTTUPUS2.M",   # Total petroleum products supplied
}


def fetch_eia_data(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    api_key: Optional[str] = None,
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Fetch multiple EIA monthly series and return a merged DataFrame."""
    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    key = api_key or os.environ.get("EIA_API_KEY")
    if not key:
        warnings.warn(
            "EIA_API_KEY not set and no api_key argument provided. "
            "Falling back to synthetic EIA data."
        )
        return _synthetic_eia(start, end)

    import requests

    frames: list[pd.DataFrame] = []
    base_url = "https://api.eia.gov/v2/seriesid/{series_id}"

    for col_name, series_id in _EIA_SERIES.items():
        try:
            url = base_url.format(series_id=series_id)
            resp = requests.get(
                url,
                params={
                    "api_key": key,
                    "start": start[:7],   # YYYY-MM
                    "end": end[:7],
                    "frequency": "monthly",
                    "data[0]": "value",
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                    "offset": 0,
                    "length": 5000,
                },
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            records = payload.get("response", {}).get("data", [])
            if not records:
                warnings.warn(f"EIA returned no data for series {series_id}.")
                continue
            s = (
                pd.DataFrame(records)[["period", "value"]]
                .rename(columns={"period": "date", "value": col_name})
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .set_index("date")
            )
            s[col_name] = pd.to_numeric(s[col_name], errors="coerce")
            frames.append(s)
        except Exception as exc:
            warnings.warn(f"EIA fetch failed for {series_id} ({exc}); skipping.")

    if not frames:
        warnings.warn("All EIA fetches failed; falling back to synthetic EIA data.")
        return _synthetic_eia(start, end)

    df = frames[0].join(frames[1:], how="outer").sort_index()
    df.index = df.index.to_period("M").to_timestamp()
    df.index.name = "date"
    df = df[df.index >= start]
    df = df[df.index <= end]

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

    return df


# ---------------------------------------------------------------------------
# FRED — macro indicators
# ---------------------------------------------------------------------------

_FRED_SERIES = {
    "real_gdp":        "GDPC1",      # Quarterly — will be interpolated
    "usd_index":       "DTWEXBGS",   # Trade-weighted USD index (daily → monthly)
    "industrial_prod": "INDPRO",     # Industrial production index
    "fed_funds_rate":  "FEDFUNDS",   # Federal funds rate
    "cpi":             "CPIAUCSL",   # CPI all items
}


def fetch_fred_data(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Fetch FRED series via pandas_datareader and return monthly DataFrame."""
    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    try:
        import pandas_datareader.data as web

        frames: list[pd.DataFrame] = []
        for col_name, series_id in _FRED_SERIES.items():
            try:
                raw = web.DataReader(series_id, "fred", start, end)
                raw.columns = [col_name]
                raw.index = raw.index.to_period("M").to_timestamp()
                raw = raw.resample("MS").last()
                frames.append(raw)
            except Exception as exc:
                warnings.warn(f"FRED fetch failed for {series_id} ({exc}); skipping.")

        if not frames:
            raise RuntimeError("No FRED series fetched.")

        df = frames[0].join(frames[1:], how="outer")

        # Real GDP is quarterly — interpolate linearly to monthly
        if "real_gdp" in df.columns:
            df["real_gdp"] = df["real_gdp"].interpolate(method="time")

        df = df.ffill()
        df.index.name = "date"
        df = df[df.index >= start]
        df = df[df.index <= end]

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)

        return df

    except Exception as exc:
        warnings.warn(f"FRED data fetch failed ({exc}); falling back to synthetic FRED data.")
        return _synthetic_fred(start, end)


# ---------------------------------------------------------------------------
# World Bank — global demand indicators (annual → interpolated monthly)
# ---------------------------------------------------------------------------

_WB_INDICATORS = {
    "wb_gdp_current_usd":    "NY.GDP.MKTP.CD",   # GDP current USD
    "wb_gdp_growth_pct":     "NY.GDP.MKTP.KD.ZG", # GDP growth %
    "wb_gni_current_usd":    "NY.GNP.MKTP.CD",   # GNI current USD
    "wb_energy_use_kg_oil":  "EG.USE.PCAP.KG.OE", # Energy use per capita
    "wb_pop_growth_pct":     "SP.POP.GROW",       # Population growth %
}


def fetch_world_bank_data(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Fetch World Bank annual indicators and interpolate to monthly."""
    if cache_path and Path(cache_path).exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    try:
        import pandas_datareader.wb as wb

        start_year = int(start[:4])
        end_year = int(end[:4])

        frames: list[pd.DataFrame] = []
        for col_name, indicator in _WB_INDICATORS.items():
            try:
                # country="WLD" = world aggregate
                raw = wb.download(
                    indicator=indicator,
                    country="WLD",
                    start=start_year,
                    end=end_year,
                )
                # MultiIndex (country, year) → drop country level
                raw = raw.droplevel("country")
                raw.index = pd.to_datetime(raw.index.astype(str))
                raw.columns = [col_name]
                frames.append(raw)
            except Exception as exc:
                warnings.warn(f"World Bank fetch failed for {indicator} ({exc}); skipping.")

        if not frames:
            raise RuntimeError("No World Bank series fetched.")

        annual = frames[0].join(frames[1:], how="outer").sort_index()

        # Reindex to monthly and interpolate
        monthly_idx = pd.date_range(start, end, freq="MS")
        df = annual.reindex(annual.index.union(monthly_idx)).sort_index()
        df = df.interpolate(method="time").reindex(monthly_idx)
        df.index.name = "date"

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)

        return df

    except Exception as exc:
        warnings.warn(f"World Bank data fetch failed ({exc}); falling back to synthetic World Bank data.")
        return _synthetic_world_bank(start, end)


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

def build_dataset(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    eia_api_key: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Merge Brent prices, EIA supply data, FRED macro, and World Bank indicators
    into a single monthly DataFrame.  Quarterly / annual series are forward-filled
    after linear interpolation so every row is complete.
    """
    cache_dir = Path(cache_dir) if cache_dir else None

    prices = fetch_brent_prices(
        start, end,
        cache_path=cache_dir / "brent_daily.csv" if cache_dir else None,
    )
    eia = fetch_eia_data(
        start, end,
        api_key=eia_api_key,
        cache_path=cache_dir / "eia_data.csv" if cache_dir else None,
    )
    fred = fetch_fred_data(
        start, end,
        cache_path=cache_dir / "fred_data.csv" if cache_dir else None,
    )
    wb = fetch_world_bank_data(
        start, end,
        cache_path=cache_dir / "world_bank_data.csv" if cache_dir else None,
    )

    df = prices.join([eia, fred, wb], how="left")

    # Final forward-fill for any residual gaps then drop rows where price is missing
    df = df.ffill().dropna(subset=["price_brent"])
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Synthetic fallback generators (deterministic seed)
# ---------------------------------------------------------------------------

def _synthetic_brent(start: str, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    log_returns = 0.03 / 252 + (0.35 / np.sqrt(252)) * rng.standard_normal(n)
    log_returns[0] = 0.0
    prices = 28.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"price_brent": prices}, index=dates)


def _synthetic_eia(start: str, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range(start, end, freq="MS")
    n = len(dates)
    t = np.arange(n)

    base = 5800 + 25 * t
    shale_start = max(0, (pd.Timestamp("2010-01-01") - pd.Timestamp(start)).days // 30)
    shale = np.zeros(n)
    if shale_start < n:
        shale[shale_start:] = 45 * np.arange(n - shale_start)
    trend = np.minimum(base + shale, 13_500)
    covid_start = max(0, (pd.Timestamp("2020-03-01") - pd.Timestamp(start)).days // 30)
    covid_end = min(n, covid_start + 7)
    if covid_start < n:
        trend[covid_start:covid_end] *= np.linspace(0.82, 0.92, covid_end - covid_start)

    df = pd.DataFrame(
        {
            "production_kbpd":       np.maximum(trend + rng.normal(0, 120, n), 4000),
            "inventory_commercial":  400_000 + 20_000 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 5000, n),
            "inventory_spr":         700_000 - 500 * t + rng.normal(0, 2000, n),
            "imports_kbpd":          7000 - 10 * t + rng.normal(0, 200, n),
            "exports_kbpd":          500 + 15 * t + rng.normal(0, 100, n),
            "refinery_util_pct":     90 + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.5, n),
            "product_supplied_kbpd": 19_000 + 50 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 300, n),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _synthetic_fred(start: str, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(start, end, freq="MS")
    n = len(dates)
    t = np.arange(n)
    df = pd.DataFrame(
        {
            "real_gdp":        18_000 + 80 * t + rng.normal(0, 200, n),
            "usd_index":       100 + 5 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 2, n),
            "industrial_prod": 100 + 0.1 * t + rng.normal(0, 1.5, n),
            "fed_funds_rate":  np.clip(2.0 + 0.05 * np.sin(2 * np.pi * t / 84) + rng.normal(0, 0.3, n), 0, 20),
            "cpi":             170 + 0.4 * t + rng.normal(0, 1, n),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _synthetic_world_bank(start: str, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range(start, end, freq="MS")
    n = len(dates)
    t = np.arange(n)
    df = pd.DataFrame(
        {
            "wb_gdp_current_usd":   (70e12 + 1e12 * t / 12) * (1 + rng.normal(0, 0.005, n)),
            "wb_gdp_growth_pct":    2.5 + rng.normal(0, 1.0, n),
            "wb_gni_current_usd":   (68e12 + 9e11 * t / 12) * (1 + rng.normal(0, 0.005, n)),
            "wb_energy_use_kg_oil": 1900 - 0.5 * t / 12 + rng.normal(0, 20, n),
            "wb_pop_growth_pct":    1.1 - 0.003 * t / 12 + rng.normal(0, 0.05, n),
        },
        index=dates,
    )
    df.index.name = "date"
    return df
