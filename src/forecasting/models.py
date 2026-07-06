import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ForecastModel:
    def fit(self, ts: pd.Series):
        pass
        
    def predict(self, steps: int) -> pd.Series:
        pass

class BaselineModel(ForecastModel):
    """
    Seasonal Naive baseline model.
    Predicts that demand for day t is the same as demand for day t - season_length.
    """
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.history = None
        
    def fit(self, ts: pd.Series):
        self.history = ts.copy()
        
    def predict(self, steps: int) -> pd.Series:
        if self.history is None or len(self.history) < self.season_length:
            raise ValueError("Not enough history to predict with seasonal naive.")
            
        preds = []
        hist_len = len(self.history)
        for i in range(steps):
            # For each step ahead, look back to the corresponding day in the history
            lookback_idx = hist_len - self.season_length + (i % self.season_length)
            preds.append(self.history.iloc[lookback_idx])
            
        return pd.Series(preds, index=pd.date_range(start=self.history.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D'))

class AdvancedModel(ForecastModel):
    """
    Holt-Winters Exponential Smoothing model.
    Captures trend and seasonality for more robust forecasting.
    """
    def __init__(self, seasonal_periods: int = 7, trend: str = 'add', seasonal: str = 'add'):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        
    def fit(self, ts: pd.Series):
        # We ensure frequency is set to 'D' for daily data
        ts = ts.asfreq('D').ffill().fillna(0)
        self.model = ExponentialSmoothing(
            ts, 
            seasonal_periods=self.seasonal_periods, 
            trend=self.trend, 
            seasonal=self.seasonal,
            initialization_method="estimated"
        ).fit()
        
    def predict(self, steps: int) -> pd.Series:
        if self.model is None:
            raise ValueError("Model must be fit before predicting.")
        
        preds = self.model.forecast(steps)
        # Never forecast negative demand/revenue
        preds = preds.clip(lower=0)
        return preds
