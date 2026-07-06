import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sqlalchemy import text
import logging

from src.storage.sql.database import DatabaseClient
from src.forecasting.models import BaselineModel, AdvancedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_weekly_revenue_data() -> pd.Series:
    client = DatabaseClient()
    query = text("""
        SELECT date, SUM(amount) as daily_revenue 
        FROM financial_transactions 
        WHERE transaction_type = 'Revenue'
        GROUP BY date
        ORDER BY date ASC
    """)
    with client.get_session() as session:
        result = session.execute(query).fetchall()
        
    df = pd.DataFrame(result, columns=['date', 'daily_revenue'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 1. Fill missing days with 0 to ensure continuous daily series
    df = df.asfreq('D').fillna(0)
    
    # 2. TRUNCATE AT 2017-09-30
    # The raw DataCo dataset contains a massive synthetic data injection starting Oct 3, 2017
    # where order volume drops 80% and artificially oscillates exactly between 68 and 69 orders/day.
    # We truncate the evaluation to September 2017 to maintain a valid, stable regime for time-series backtesting.
    df = df.loc[:'2017-09-30']
    
    # 3. Aggregate to Weekly (Mondays)
    weekly_ts = df['daily_revenue'].resample('W-MON').sum()
    return weekly_ts

def run_evaluation():
    logger.info("Starting forecasting evaluation (Stable Regime: Pre-Oct 2017)...")
    ts = get_weekly_revenue_data()
    
    if len(ts) < 20:
        logger.error("Not enough data to run meaningful time-series evaluation.")
        return
        
    # Strict chronological split
    # Forecast Horizon & Test Set: 4 Weeks
    forecast_horizon = 4
    train_size = len(ts) - forecast_horizon
    
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]
    
    logger.info(f"Target Variable: Weekly Revenue")
    logger.info(f"Aggregation: Weekly (W-MON)")
    logger.info(f"Training Start: {train.index.min().date()}, End: {train.index.max().date()} ({len(train)} weeks)")
    logger.info(f"Test Start: {test.index.min().date()}, End: {test.index.max().date()} ({len(test)} weeks)")
    
    # Baseline (Seasonal Naive) - 4 weeks = ~1 month seasonality
    baseline = BaselineModel(season_length=4)
    baseline.fit(train)
    baseline_preds = baseline.predict(steps=len(test))
    
    # Advanced Model (Holt-Winters)
    advanced = AdvancedModel(seasonal_periods=4, trend='add', seasonal='add')
    advanced.fit(train)
    advanced_preds = advanced.predict(steps=len(test))
    
    # Calculate Metrics
    # Baseline
    b_mae = mean_absolute_error(test, baseline_preds)
    b_rmse = root_mean_squared_error(test, baseline_preds)
    b_mape = mean_absolute_percentage_error(test, baseline_preds)
    
    # Advanced
    a_mae = mean_absolute_error(test, advanced_preds)
    a_rmse = root_mean_squared_error(test, advanced_preds)
    a_mape = mean_absolute_percentage_error(test, advanced_preds)
    
    logger.info(f"--- RESULTS ---")
    logger.info(f"Baseline (Seasonal Naive):")
    logger.info(f"  MAE  = {b_mae:.2f}")
    logger.info(f"  RMSE = {b_rmse:.2f}")
    logger.info(f"  MAPE = {b_mape*100:.2f}%")
    
    logger.info(f"Advanced (Holt-Winters):")
    logger.info(f"  MAE  = {a_mae:.2f}")
    logger.info(f"  RMSE = {a_rmse:.2f}")
    logger.info(f"  MAPE = {a_mape*100:.2f}%")
    
    if b_mae < a_mae:
        logger.info("WINNER: Baseline (Seasonal Naive). The naive baseline outperforms Holt-Winters on the stable weekly regime.")
    else:
        logger.info("WINNER: Advanced (Holt-Winters).")
    
    return {
        "baseline_mae": b_mae,
        "baseline_mape": b_mape,
        "advanced_mae": a_mae,
        "advanced_mape": a_mape
    }

if __name__ == "__main__":
    run_evaluation()
