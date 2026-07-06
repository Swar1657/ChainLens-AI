import pandas as pd
import numpy as np

def detect_anomalies_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detects anomalies in a specific numeric column using the Z-score method.
    Returns the dataframe with an 'is_anomaly' boolean column.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()
        
    mean = df[column].mean()
    std = df[column].std()
    
    if pd.isna(std) or std == 0:
        df['is_anomaly'] = False
        return df
        
    df['z_score'] = (df[column] - mean) / std
    df['is_anomaly'] = np.abs(df['z_score']) > threshold
    return df

def detect_anomalies_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Detects anomalies in a specific numeric column using the IQR method.
    Returns the dataframe with an 'is_anomaly' boolean column.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame()
        
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    df['is_anomaly'] = (df[column] < lower_bound) | (df[column] > upper_bound)
    return df
