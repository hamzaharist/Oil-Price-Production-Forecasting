# ============================================================
#  HOW THE ML PIPELINE WORKS — step-by-step walkthrough
#  Run: python notebooks/how_it_works.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from src.data_ingestion import build_dataset
from src.feature_engineering import engineer_features
from src.modeling import RandomForestModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEP = "\n" + "=" * 60 + "\n"


# ─────────────────────────────────────────────
# STEP 1 — Load data from APIs
# ─────────────────────────────────────────────
print(SEP + "STEP 1: LOAD RAW DATA FROM APIs")

df_raw = build_dataset(
    start="2000-01-01",
    end="2025-12-31",
    cache_dir="data/",
)

print(f"Shape          : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"Date range     : {df_raw.index[0].date()}  ->  {df_raw.index[-1].date()}")
print(f"Columns        : {list(df_raw.columns)}")
print("\nFirst 3 rows:")
print(df_raw.head(3).to_string())


# ─────────────────────────────────────────────
# STEP 2 — Engineer features
# ─────────────────────────────────────────────
print(SEP + "STEP 2: ENGINEER FEATURES")

df_feat = engineer_features(df_raw, target_col="price_brent")

feature_cols = [c for c in df_feat.columns if c != "price_brent"]

print(f"Shape after engineering : {df_feat.shape[0]} rows × {df_feat.shape[1]} columns")
print(f"Total features          : {len(feature_cols)}")
print(f"\nAll feature names:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:>2}. {col}")


# ─────────────────────────────────────────────
# STEP 3 — What the model sees (one sample row)
# ─────────────────────────────────────────────
print(SEP + "STEP 3: WHAT THE MODEL SEES FOR A SINGLE MONTH")

sample_date = df_feat.index[-24]   # pick a row 24 months before the end
sample = df_feat.loc[sample_date]
actual_price = sample["price_brent"]

print(f"Month being predicted : {sample_date.strftime('%B %Y')}")
print(f"Actual Brent price    : ${actual_price:.2f} / bbl")
print(f"\nInput features the model receives:")
for col in feature_cols:
    print(f"  {col:<35} = {sample[col]:.4f}")


# ─────────────────────────────────────────────
# STEP 4 — Train / test split
# ─────────────────────────────────────────────
print(SEP + "STEP 4: TRAIN / TEST SPLIT (80% train, 20% test)")

# Time-based split — never shuffle time series data
split_idx = int(len(df_feat) * 0.80)
df_train = df_feat.iloc[:split_idx]
df_test  = df_feat.iloc[split_idx:]

print(f"Training set : {df_train.index[0].date()}  ->  {df_train.index[-1].date()}  ({len(df_train)} months)")
print(f"Test set     : {df_test.index[0].date()}   ->  {df_test.index[-1].date()}  ({len(df_test)} months)")


# ─────────────────────────────────────────────
# STEP 5 — Train the Random Forest
# ─────────────────────────────────────────────
print(SEP + "STEP 5: TRAIN RANDOM FOREST (200 trees)")

model = RandomForestModel(feature_cols=feature_cols, target_col="price_brent", n_estimators=200)
model.fit(df_train)

print("Training complete.")
print(f"  Trees built         : 200")
print(f"  Features used       : {len(feature_cols)}")
print(f"  Training samples    : {len(df_train)}")


# ─────────────────────────────────────────────
# STEP 6 — Make predictions on the test set
# ─────────────────────────────────────────────
print(SEP + "STEP 6: PREDICTIONS ON TEST SET")

preds = model.predict(df_test)
actuals = df_test["price_brent"]

mae  = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
mape = (np.abs((actuals - preds) / actuals) * 100).mean()

print(f"{'Month':<15} {'Actual':>10} {'Predicted':>12} {'Error':>10}")
print("-" * 50)
for date, actual, pred in zip(actuals.index[:15], actuals.values[:15], preds.values[:15]):
    error = pred - actual
    print(f"{date.strftime('%b %Y'):<15} ${actual:>8.2f}   ${pred:>9.2f}   {error:>+8.2f}")
print(f"{'... (showing first 15 of ' + str(len(actuals)) + ' test months)':}")

print(f"\nTest Set Accuracy:")
print(f"  MAE  (avg error)      : ${mae:.2f}")
print(f"  RMSE (penalises big)  : ${rmse:.2f}")
print(f"  MAPE (% off on avg)   : {mape:.1f}%")


# ─────────────────────────────────────────────
# STEP 7 — Feature importances
# ─────────────────────────────────────────────
print(SEP + "STEP 7: WHICH VARIABLES MATTER MOST? (MDI Feature Importance)")

importances = model.feature_importances(top_n=15)

print(f"{'Rank':<6} {'Feature':<35} {'Importance':>12}  {'Bar'}")
print("-" * 75)
for rank, (feat, score) in enumerate(importances.items(), 1):
    bar = "#" * int(score * 400)
    print(f"{rank:<6} {feat:<35} {score:>11.4f}  {bar}")

print("\nInterpretation:")
print("  Higher importance = that variable has more influence on the predicted price.")
print("  e.g. if 'price_brent_lag_1' is #1, last month's price is the strongest signal.")


# ─────────────────────────────────────────────
# STEP 8 — Predict next month
# ─────────────────────────────────────────────
print(SEP + "STEP 8: PREDICT NEXT MONTH'S PRICE")

# Retrain on ALL available data, then predict from the last row
model_full = RandomForestModel(feature_cols=feature_cols, target_col="price_brent", n_estimators=200)
model_full.fit(df_feat)

last_row = df_feat.iloc[[-1]]
next_pred = model_full.predict(last_row).iloc[0]
last_actual = df_feat["price_brent"].iloc[-1]
last_date = df_feat.index[-1]

next_month = last_date + pd.DateOffset(months=1)

print(f"Last known month : {last_date.strftime('%B %Y')}  ->  ${last_actual:.2f} / bbl")
print(f"Predicted next   : {next_month.strftime('%B %Y')}  ->  ${next_pred:.2f} / bbl")
print(f"Direction        : {'UP ^' if next_pred > last_actual else 'DOWN v'}  ({next_pred - last_actual:+.2f} USD)")

