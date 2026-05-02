# Evaluation — rolling-window backtest and error metrics

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# --- Metrics ---

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# --- Backtest result container ---

@dataclass
class BacktestResult:
    model_name: str
    actuals: list[float] = field(default_factory=list)
    predictions: list[float] = field(default_factory=list)
    dates: list = field(default_factory=list)

    @property
    def rmse(self):
        return rmse(np.array(self.actuals), np.array(self.predictions))

    @property
    def mape(self):
        return mape(np.array(self.actuals), np.array(self.predictions))

    def summary(self):
        return {
            "model": self.model_name,
            "RMSE": round(self.rmse, 4),
            "MAPE (%)": round(self.mape, 2),
            "n_folds": len(self.actuals),
        }

    def to_dataframe(self):
        return pd.DataFrame(
            {"date": self.dates, "actual": self.actuals, "predicted": self.predictions}
        ).set_index("date")


# --- Rolling backtests ---

def rolling_backtest_arima(series, order=(2, 1, 2), min_train=60, horizon=1, step=1):
    from src.modeling import ARIMAModel
    result = BacktestResult(model_name=f"ARIMA{order}")
    n = len(series)
    for end in range(min_train, n - horizon + 1, step):
        train = series.iloc[:end]
        actual = series.iloc[end + horizon - 1]
        model = ARIMAModel(order=order)
        model.fit(train)
        pred = model.predict(steps=horizon).iloc[-1]
        result.actuals.append(float(actual))
        result.predictions.append(float(pred))
        result.dates.append(series.index[end + horizon - 1])
    return result


def rolling_backtest_prophet(series, min_train=60, horizon=1, step=1):
    from src.modeling import ProphetModel
    result = BacktestResult(model_name="Prophet")
    n = len(series)
    for end in range(min_train, n - horizon + 1, step):
        train = series.iloc[:end]
        actual = series.iloc[end + horizon - 1]
        model = ProphetModel()
        model.fit(train)
        pred = model.predict(steps=horizon).iloc[-1]
        result.actuals.append(float(actual))
        result.predictions.append(float(pred))
        result.dates.append(series.index[end + horizon - 1])
    return result


def rolling_backtest_rf(df, target_col="price_brent", feature_cols=None,
                        min_train=60, horizon=1, step=1):
    from src.modeling import RandomForestModel
    result = BacktestResult(model_name="RandomForest")
    n = len(df)
    for end in range(min_train, n - horizon + 1, step):
        train = df.iloc[:end]
        test_row = df.iloc[[end + horizon - 1]]
        actual = df[target_col].iloc[end + horizon - 1]
        model = RandomForestModel(feature_cols=feature_cols, target_col=target_col)
        model.fit(train)
        pred = model.predict(test_row).iloc[0]
        result.actuals.append(float(actual))
        result.predictions.append(float(pred))
        result.dates.append(df.index[end + horizon - 1])
    return result


# --- Comparison ---

def compare_models(results):
    rows = [r.summary() for r in results]
    return pd.DataFrame(rows).set_index("model")
