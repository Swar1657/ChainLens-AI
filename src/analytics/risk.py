import pandas as pd

def evaluate_delivery_risk(days_real: int, days_scheduled: int) -> dict:
    """
    Evaluates risk for a single delivery based on scheduled vs actual days.
    Returns a severity and a continuous risk score.
    """
    delay = days_real - days_scheduled
    if delay <= 0:
        return {"severity": "SAFE", "score": 0.0, "delay_days": delay}
    elif delay <= 2:
        return {"severity": "WARNING", "score": 0.5, "delay_days": delay}
    else:
        return {"severity": "CRITICAL", "score": 1.0, "delay_days": delay}

def evaluate_inventory_risk(current_stock: int, reorder_point: int) -> dict:
    """
    Evaluates inventory risk based on distance to reorder point.
    """
    if current_stock <= 0:
        return {"severity": "STOCKOUT", "score": 1.0, "stock_gap": current_stock - reorder_point}
    elif current_stock <= reorder_point:
        return {"severity": "CRITICAL", "score": 0.8, "stock_gap": current_stock - reorder_point}
    elif current_stock <= reorder_point * 1.5:
        return {"severity": "WARNING", "score": 0.4, "stock_gap": current_stock - reorder_point}
    else:
        return {"severity": "SAFE", "score": 0.0, "stock_gap": current_stock - reorder_point}

def summarize_portfolio_risk(shipments_df: pd.DataFrame, inventory_df: pd.DataFrame) -> dict:
    """
    Calculates aggregated portfolio risk from dataframes.
    shipments_df expected columns: 'days_for_shipping_real', 'days_for_shipping_scheduled'
    inventory_df expected columns: 'stock_level', 'reorder_point'
    """
    late_rate = 0.0
    if not shipments_df.empty:
        late_shipments = shipments_df[shipments_df['days_for_shipping_real'] > shipments_df['days_for_shipping_scheduled']]
        late_rate = len(late_shipments) / len(shipments_df)
        
    stockout_rate = 0.0
    critical_stock_rate = 0.0
    if not inventory_df.empty:
        stockouts = inventory_df[inventory_df['stock_level'] <= 0]
        critical = inventory_df[inventory_df['stock_level'] <= inventory_df['reorder_point']]
        
        stockout_rate = len(stockouts) / len(inventory_df)
        critical_stock_rate = len(critical) / len(inventory_df)
        
    return {
        "late_shipment_rate": round(late_rate, 4),
        "stockout_rate": round(stockout_rate, 4),
        "critical_inventory_rate": round(critical_stock_rate, 4)
    }
