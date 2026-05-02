# Export forecasting data to Power BI-ready CSVs
# Run: python export_for_powerbi.py

import sys, os, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.data_ingestion import build_dataset
from src.feature_engineering import engineer_features
from src.evaluation import (
    rolling_backtest_arima,
    rolling_backtest_prophet,
    rolling_backtest_rf,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "powerbi_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

print("=" * 60)
print("  POWER BI DATA EXPORT")
print("=" * 60)

# 1. Raw dataset
print("\n[1/5] Building raw dataset ...")
df_raw = build_dataset(start="2000-01-01", end="2025-12-31", cache_dir=DATA_DIR)
df_raw.index.name = "date"

export_raw = df_raw.copy()
export_raw.insert(0, "date", export_raw.index)
export_raw.reset_index(drop=True, inplace=True)
export_raw.to_csv(os.path.join(OUTPUT_DIR, "01_raw_data.csv"), index=False)
print(f"   -> 01_raw_data.csv  ({len(export_raw)} rows)")

# 2. Feature-engineered dataset
print("[2/5] Engineering features ...")
df_feat = engineer_features(df_raw, target_col="price_brent")

export_feat = df_feat.copy()
export_feat.insert(0, "date", export_feat.index)
export_feat.reset_index(drop=True, inplace=True)
export_feat.to_csv(os.path.join(OUTPUT_DIR, "02_features.csv"), index=False)
print(f"   -> 02_features.csv  ({len(export_feat)} rows, {len(export_feat.columns)} cols)")

# 3. Backtest results
target = "price_brent"
series = df_raw[target].copy()

print("[3/5] Running ARIMA backtest ...")
res_arima = rolling_backtest_arima(series, order=(2, 1, 2), min_train=60, horizon=1, step=3)
bt_arima = res_arima.to_dataframe()
bt_arima["model"] = "ARIMA(2,1,2)"
bt_arima["residual"] = bt_arima["actual"] - bt_arima["predicted"]
bt_arima["abs_pct_error"] = abs(bt_arima["residual"]) / bt_arima["actual"] * 100

print("[3/5] Running Prophet backtest ...")
res_prophet = rolling_backtest_prophet(series, min_train=60, horizon=1, step=3)
bt_prophet = res_prophet.to_dataframe()
bt_prophet["model"] = "Prophet"
bt_prophet["residual"] = bt_prophet["actual"] - bt_prophet["predicted"]
bt_prophet["abs_pct_error"] = abs(bt_prophet["residual"]) / bt_prophet["actual"] * 100

print("[3/5] Running Random Forest backtest ...")
feature_cols = [c for c in df_feat.columns if c != target]
res_rf = rolling_backtest_rf(df_feat, target_col=target, feature_cols=feature_cols,
                             min_train=60, horizon=1, step=3)
bt_rf = res_rf.to_dataframe()
bt_rf["model"] = "RandomForest"
bt_rf["residual"] = bt_rf["actual"] - bt_rf["predicted"]
bt_rf["abs_pct_error"] = abs(bt_rf["residual"]) / bt_rf["actual"] * 100

bt_all = pd.concat([bt_arima, bt_prophet, bt_rf], ignore_index=False)
bt_all.insert(0, "date", bt_all.index)
bt_all.reset_index(drop=True, inplace=True)
bt_all.to_csv(os.path.join(OUTPUT_DIR, "03_backtest_results.csv"), index=False)
print(f"   -> 03_backtest_results.csv  ({len(bt_all)} rows)")

# 4. Model summary metrics
print("[4/5] Computing summary metrics ...")
summary_rows = []
for name, res in [("ARIMA(2,1,2)", res_arima), ("Prophet", res_prophet), ("RandomForest", res_rf)]:
    summary_rows.append({
        "model": name,
        "RMSE": round(res.rmse, 4),
        "MAPE_pct": round(res.mape, 2),
        "n_folds": len(res.actuals),
        "mean_actual": round(np.mean(res.actuals), 2),
        "mean_predicted": round(np.mean(res.predictions), 2),
        "std_residual": round(np.std(np.array(res.actuals) - np.array(res.predictions)), 2),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUTPUT_DIR, "04_model_summary.csv"), index=False)
print(f"   -> 04_model_summary.csv  ({len(df_summary)} rows)")

# 5. Feature importances
print("[5/5] Extracting RF feature importances ...")
from src.modeling import RandomForestModel

rf_full = RandomForestModel(feature_cols=feature_cols, target_col=target)
rf_full.fit(df_feat)
imp = rf_full.feature_importances(top_n=20)

df_imp = imp.reset_index()
df_imp.columns = ["feature", "importance"]
df_imp["rank"] = range(1, len(df_imp) + 1)
df_imp.to_csv(os.path.join(OUTPUT_DIR, "05_feature_importances.csv"), index=False)
print(f"   -> 05_feature_importances.csv  ({len(df_imp)} rows)")

print("\n" + "=" * 60)
print(f"  EXPORT COMPLETE -> {OUTPUT_DIR}")
print("=" * 60)
