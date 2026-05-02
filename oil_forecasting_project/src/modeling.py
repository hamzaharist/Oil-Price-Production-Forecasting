# Modeling — ARIMA, Prophet, and Random Forest wrappers

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# --- ARIMA ---

class ARIMAModel:
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self._model = None
        self._result = None

    def fit(self, y_train):
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = ARIMA(y_train, order=self.order)
            self._result = self._model.fit()
        return self

    def predict(self, steps):
        return self._result.forecast(steps=steps)

    @property
    def name(self):
        return f"ARIMA{self.order}"


# --- Prophet ---

class ProphetModel:
    def __init__(self, yearly_seasonality=True, **kwargs):
        self.yearly_seasonality = yearly_seasonality
        self._kwargs = kwargs
        self._model = None

    def fit(self, y_train):
        from prophet import Prophet
        prophet_df = y_train.reset_index()
        prophet_df.columns = ["ds", "y"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(yearly_seasonality=self.yearly_seasonality, **self._kwargs)
            m.fit(prophet_df)
        self._model = m
        return self

    def predict(self, steps):
        future = self._model.make_future_dataframe(periods=steps, freq="MS")
        fc = self._model.predict(future)
        preds = fc.set_index("ds")["yhat"].iloc[-steps:]
        preds.index.name = "date"
        return preds

    @property
    def name(self):
        return "Prophet"


# --- Random Forest ---

class RandomForestModel:
    def __init__(self, feature_cols=None, target_col="price_brent",
                 n_estimators=200, random_state=42):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def _resolve_features(self, df):
        if self.feature_cols:
            return self.feature_cols
        return [c for c in df.columns if c != self.target_col]

    def fit(self, df_train):
        feats = self._resolve_features(df_train)
        self._rf.fit(df_train[feats].values, df_train[self.target_col].values)
        self._features = feats
        return self

    def predict(self, df_test):
        preds = self._rf.predict(df_test[self._features].values)
        return pd.Series(preds, index=df_test.index, name="prediction")

    def feature_importances(self, top_n=10):
        imp = pd.Series(self._rf.feature_importances_, index=self._features)
        return imp.sort_values(ascending=False).head(top_n)

    @property
    def name(self):
        return "RandomForest"
