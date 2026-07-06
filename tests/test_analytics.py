import pandas as pd
from src.analytics.finance import calculate_period_metrics, calculate_period_over_period
from src.analytics.risk import evaluate_delivery_risk, evaluate_inventory_risk, summarize_portfolio_risk
from src.analytics.anomaly import detect_anomalies_zscore, detect_anomalies_iqr

def test_finance_metrics():
    df = pd.DataFrame([
        {"date": "2023-01-01", "transaction_type": "Revenue", "amount": 1000},
        {"date": "2023-01-01", "transaction_type": "COGS", "amount": 600},
        {"date": "2023-01-01", "transaction_type": "OpEx", "amount": 100},
    ])
    metrics = calculate_period_metrics(df)
    assert metrics['total_revenue'] == 1000
    assert metrics['gross_profit'] == 400
    assert metrics['net_profit'] == 300
    assert metrics['gross_margin_pct'] == 40.0
    assert metrics['net_margin_pct'] == 30.0

def test_pop_metrics():
    current = {"total_revenue": 1100, "net_profit": 330}
    previous = {"total_revenue": 1000, "net_profit": 300}
    trends = calculate_period_over_period(current, previous)
    assert trends['total_revenue_growth_pct'] == 10.0
    assert trends['net_profit_growth_pct'] == 10.0

def test_risk_metrics():
    # Delivery
    safe = evaluate_delivery_risk(5, 5)
    assert safe['severity'] == 'SAFE'
    
    crit = evaluate_delivery_risk(8, 5)
    assert crit['severity'] == 'CRITICAL'
    
    # Inventory
    stockout = evaluate_inventory_risk(0, 20)
    assert stockout['severity'] == 'STOCKOUT'
    
    safe_inv = evaluate_inventory_risk(100, 20)
    assert safe_inv['severity'] == 'SAFE'

def test_portfolio_risk():
    shipments = pd.DataFrame([
        {"days_for_shipping_real": 5, "days_for_shipping_scheduled": 5},
        {"days_for_shipping_real": 6, "days_for_shipping_scheduled": 5}
    ])
    inv = pd.DataFrame([
        {"stock_level": 50, "reorder_point": 20},
        {"stock_level": 0, "reorder_point": 20}
    ])
    
    portfolio = summarize_portfolio_risk(shipments, inv)
    assert portfolio['late_shipment_rate'] == 0.5
    assert portfolio['stockout_rate'] == 0.5

def test_anomaly_detection():
    df = pd.DataFrame({
        "revenue": [100, 110, 105, 95, 1000, 102] # 1000 is an outlier
    })
    # Z-score
    res_z = detect_anomalies_zscore(df.copy(), "revenue", threshold=2.0)
    assert res_z.iloc[4]['is_anomaly'] == True
    assert res_z.iloc[0]['is_anomaly'] == False
    
    # IQR
    res_iqr = detect_anomalies_iqr(df.copy(), "revenue")
    assert res_iqr.iloc[4]['is_anomaly'] == True
    assert res_iqr.iloc[0]['is_anomaly'] == False
