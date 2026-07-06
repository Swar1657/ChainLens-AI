import pandas as pd
import numpy as np
from src.forecasting.models import BaselineModel, AdvancedModel

def test_baseline_forecasting():
    # Simple seasonal pattern: [10, 20, 10, 20, 10, 20]
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    ts = pd.Series([10, 20, 10, 20, 10, 20], index=dates)
    
    model = BaselineModel(season_length=2)
    model.fit(ts)
    
    preds = model.predict(steps=2)
    assert len(preds) == 2
    assert preds.iloc[0] == 10
    assert preds.iloc[1] == 20

def test_advanced_forecasting():
    dates = pd.date_range("2023-01-01", periods=14, freq="D")
    # Linear trend + noise
    ts = pd.Series([i + np.random.normal(0, 0.1) for i in range(14)], index=dates)
    
    model = AdvancedModel(seasonal_periods=7, trend='add', seasonal=None)
    model.fit(ts)
    
    preds = model.predict(steps=3)
    assert len(preds) == 3
    # Just asserting it doesn't crash and outputs correct shape
    assert preds.index[0] == pd.Timestamp("2023-01-15")
