# Feature engineering — lag, rolling, calendar, and domain features

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(df, target_col="price_brent", lags=None):
    # Add lagged versions of target column
    if lags is None:
        lags = [1, 2, 3, 6, 12]
    df = df.copy()
    for k in lags:
        df[f"{target_col}_lag_{k}"] = df[target_col].shift(k)
    return df


def add_rolling_features(df, target_col="price_brent", windows=None):
    # Add rolling mean and std
    if windows is None:
        windows = [3, 6, 12]
    df = df.copy()
    for w in windows:
        df[f"{target_col}_rmean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_rstd_{w}"] = df[target_col].rolling(w).std()
    return df


def add_calendar_features(df):
    # Add cyclical month encoding and year
    df = df.copy()
    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["year"] = df.index.year
    return df


def add_derived_features(df, price_col="price_brent", prod_col="production_kbpd"):
    # Add YoY change and production-to-price ratio
    df = df.copy()
    df["yoy_pct_change"] = df[price_col].pct_change(12) * 100
    if prod_col in df.columns:
        df["prod_price_ratio"] = df[prod_col] / df[price_col]
    return df


def engineer_features(df, target_col="price_brent", drop_na=True):
    # Run full feature pipeline
    out = df.copy()
    out = add_lag_features(out, target_col)
    out = add_rolling_features(out, target_col)
    out = add_calendar_features(out)
    out = add_derived_features(out, target_col)
    if drop_na:
        out = out.dropna()
    return out
