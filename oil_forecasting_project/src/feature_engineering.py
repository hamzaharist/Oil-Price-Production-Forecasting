# Feature engineering — lag, rolling, calendar, domain, and macro features

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(df, target_col="price_brent", lags=None):
    """Add lagged versions of target column."""
    if lags is None:
        lags = [1, 2, 3, 6, 12]
    df = df.copy()
    for k in lags:
        df[f"{target_col}_lag_{k}"] = df[target_col].shift(k)
    return df


def add_rolling_features(df, target_col="price_brent", windows=None):
    """Add rolling mean and std."""
    if windows is None:
        windows = [3, 6, 12]
    df = df.copy()
    for w in windows:
        df[f"{target_col}_rmean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_rstd_{w}"] = df[target_col].rolling(w).std()
    return df


def add_calendar_features(df):
    """Add cyclical month encoding and year."""
    df = df.copy()
    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["year"] = df.index.year
    return df


def add_derived_features(df, price_col="price_brent", prod_col="production_kbpd"):
    """Add YoY change and production-to-price ratio."""
    df = df.copy()
    df["yoy_pct_change"] = df[price_col].pct_change(12) * 100
    if prod_col in df.columns:
        df["prod_price_ratio"] = df[prod_col] / df[price_col]
    return df


def add_macro_features(df):
    """Compute domain-specific macro features from the enriched dataset.

    Requires columns produced by the new data_ingestion functions:
      - product_supplied_kbpd  (EIA)
      - wb_gdp_growth_pct / wb_gni_current_usd  (World Bank)
      - inventory_commercial / inventory_spr  (EIA)
      - cpi  (FRED)
      - price_brent  (Brent spot)
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Inventory shocks — MoM absolute and percentage change
    # ------------------------------------------------------------------
    if "inventory_commercial" in df.columns:
        df["inventory_commercial_mom"] = df["inventory_commercial"].diff()
        df["inventory_commercial_mom_pct"] = df["inventory_commercial"].pct_change() * 100

    if "inventory_spr" in df.columns:
        df["inventory_spr_mom"] = df["inventory_spr"].diff()

    # Combined total inventories
    if {"inventory_commercial", "inventory_spr"}.issubset(df.columns):
        df["inventory_total"] = df["inventory_commercial"] + df["inventory_spr"]
        df["inventory_total_mom"] = df["inventory_total"].diff()

    # ------------------------------------------------------------------
    # 2. Real (inflation-adjusted) Brent price  — price / (CPI / 100)
    # ------------------------------------------------------------------
    if "cpi" in df.columns and "price_brent" in df.columns:
        df["price_brent_real"] = df["price_brent"] / (df["cpi"] / 100)

    # ------------------------------------------------------------------
    # 3. Income elasticity of demand proxy
    #    = (YoY % Δ product supplied) / (YoY % Δ global GDP)
    #    Uses wb_gdp_growth_pct if available, else falls back to
    #    wb_gni_current_usd YoY %.
    # ------------------------------------------------------------------
    if "product_supplied_kbpd" in df.columns:
        ps_yoy = df["product_supplied_kbpd"].pct_change(12) * 100
        df["product_supplied_yoy_pct"] = ps_yoy

        if "wb_gdp_growth_pct" in df.columns:
            gdp_growth = df["wb_gdp_growth_pct"]
        elif "wb_gni_current_usd" in df.columns:
            gdp_growth = df["wb_gni_current_usd"].pct_change(12) * 100
        else:
            gdp_growth = None

        if gdp_growth is not None:
            # Avoid division by zero / near-zero
            safe_gdp = gdp_growth.replace(0, np.nan)
            df["income_elasticity_proxy"] = ps_yoy / safe_gdp

    # ------------------------------------------------------------------
    # 4. Supply-demand balance proxy  (net exports + inventory build)
    # ------------------------------------------------------------------
    if {"imports_kbpd", "exports_kbpd"}.issubset(df.columns):
        df["net_exports_kbpd"] = df["exports_kbpd"] - df["imports_kbpd"]

    # ------------------------------------------------------------------
    # 5. Refinery tightness — product supplied relative to refinery util
    # ------------------------------------------------------------------
    if {"product_supplied_kbpd", "refinery_util_pct"}.issubset(df.columns):
        df["demand_per_refinery_util"] = df["product_supplied_kbpd"] / df["refinery_util_pct"]

    return df


def engineer_features(df, target_col="price_brent", drop_na=True):
    """Run full feature pipeline including new macro features."""
    out = df.copy()
    out = add_lag_features(out, target_col)
    out = add_rolling_features(out, target_col)
    out = add_calendar_features(out)
    out = add_derived_features(out, target_col)
    out = add_macro_features(out)
    if drop_na:
        out = out.dropna()
    return out
