# End-to-end oil price forecasting workflow
# Run: python -m notebooks.01_end_to_end_workflow

# %% Imports & setup
import sys, os, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

from src.data_ingestion import build_dataset
from src.feature_engineering import engineer_features
from src.evaluation import (
    rolling_backtest_arima,
    rolling_backtest_prophet,
    rolling_backtest_rf,
    compare_models,
)

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# %% 1. Data ingestion
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
df_raw = build_dataset(start="2000-01-01", end="2025-12-31", cache_dir=DATA_DIR)
print(f"Dataset shape : {df_raw.shape}")
print(f"Date range    : {df_raw.index.min().date()} to {df_raw.index.max().date()}")
print(df_raw.describe().round(2))

# %% Raw data plot
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(df_raw.index, df_raw["price_brent"], color="#1f77b4")
axes[0].set_ylabel("Brent Price (USD/bbl)")
axes[0].set_title("Monthly Brent Crude Price")
axes[1].plot(df_raw.index, df_raw["production_kbpd"], color="#d62728")
axes[1].set_ylabel("Production (kbpd)")
axes[1].set_title("U.S. Crude-Oil Production")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_raw_data.png"), dpi=150)
plt.close(fig)
print("[OK] Saved figures/01_raw_data.png")

# %% 2. Feature engineering
df_feat = engineer_features(df_raw, target_col="price_brent")
print(f"Feature matrix shape: {df_feat.shape}")
print("Columns:", list(df_feat.columns))

# %% Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr = df_feat.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "02_correlation.png"), dpi=150)
plt.close(fig)
print("[OK] Saved figures/02_correlation.png")

# %% 3. Rolling-window backtests
target = "price_brent"
series = df_raw[target].copy()

print("Running ARIMA backtest ...")
res_arima = rolling_backtest_arima(series, order=(2, 1, 2), min_train=60, horizon=1, step=3)
print(f"  ARIMA  ->  RMSE={res_arima.rmse:.2f}  MAPE={res_arima.mape:.2f}%")

print("Running Prophet backtest ...")
res_prophet = rolling_backtest_prophet(series, min_train=60, horizon=1, step=3)
print(f"  Prophet ->  RMSE={res_prophet.rmse:.2f}  MAPE={res_prophet.mape:.2f}%")

print("Running Random Forest backtest ...")
feature_cols = [c for c in df_feat.columns if c != target]
res_rf = rolling_backtest_rf(df_feat, target_col=target, feature_cols=feature_cols,
                             min_train=60, horizon=1, step=3)
print(f"  RF     ->  RMSE={res_rf.rmse:.2f}  MAPE={res_rf.mape:.2f}%")

# %% 4. Results comparison
comparison = compare_models([res_arima, res_prophet, res_rf])
print("\n" + "=" * 50)
print("MODEL BENCHMARK (1-month-ahead, expanding window)")
print("=" * 50)
print(comparison.to_string())

# %% Actual vs predicted overlay
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
titles = [res_arima.model_name, res_prophet.model_name, res_rf.model_name]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for ax, res, title, color in zip(axes, [res_arima, res_prophet, res_rf], titles, colors):
    bt = res.to_dataframe()
    ax.plot(bt.index, bt["actual"], label="Actual", color="black", linewidth=1)
    ax.plot(bt.index, bt["predicted"], label="Predicted", color=color, linewidth=1, alpha=0.8)
    ax.set_title(f"{title}  -  RMSE={res.rmse:.2f}  |  MAPE={res.mape:.1f}%")
    ax.legend(loc="upper left")
    ax.set_ylabel("USD / bbl")

axes[-1].set_xlabel("Date")
plt.suptitle("Rolling Back-test: Actual vs Predicted", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "03_backtest_results.png"), dpi=150)
plt.close(fig)
print("[OK] Saved figures/03_backtest_results.png")

# %% Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
comparison["RMSE"].plot.barh(ax=axes[0], color=colors)
axes[0].set_title("RMSE (lower is better)")
axes[0].set_xlabel("RMSE (USD)")
comparison["MAPE (%)"].plot.barh(ax=axes[1], color=colors)
axes[1].set_title("MAPE % (lower is better)")
axes[1].set_xlabel("MAPE (%)")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "04_model_comparison.png"), dpi=150)
plt.close(fig)
print("[OK] Saved figures/04_model_comparison.png")

# %% Feature importances
from src.modeling import RandomForestModel

rf_full = RandomForestModel(feature_cols=feature_cols, target_col=target)
rf_full.fit(df_feat)
importances = rf_full.feature_importances(top_n=15)

fig, ax = plt.subplots(figsize=(8, 6))
importances.sort_values().plot.barh(ax=ax, color="#2ca02c")
ax.set_title("Random Forest - Top Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "05_feature_importances.png"), dpi=150)
plt.close(fig)
print("[OK] Saved figures/05_feature_importances.png")

# %%
print("\n[DONE] Workflow complete. Check figures/ for all plots.")
