import pandas as pd

def calculate_period_metrics(df: pd.DataFrame) -> dict:
    """
    Calculates deterministic financial metrics (Revenue, COGS, OpEx, Profit, Margins)
    from a DataFrame of financial transactions.
    
    Expected DataFrame columns: 'date', 'transaction_type', 'amount'
    """
    if df.empty:
        return {
            "total_revenue": 0.0,
            "total_cogs": 0.0,
            "total_opex": 0.0,
            "gross_profit": 0.0,
            "net_profit": 0.0,
            "gross_margin_pct": 0.0,
            "net_margin_pct": 0.0
        }
        
    revenue = df[df['transaction_type'] == 'Revenue']['amount'].sum()
    cogs = df[df['transaction_type'] == 'COGS']['amount'].sum()
    opex = df[df['transaction_type'] == 'OpEx']['amount'].sum()
    
    gross_profit = revenue - cogs
    net_profit = gross_profit - opex
    
    gross_margin = (gross_profit / revenue) * 100 if revenue > 0 else 0.0
    net_margin = (net_profit / revenue) * 100 if revenue > 0 else 0.0
    
    return {
        "total_revenue": round(float(revenue), 2),
        "total_cogs": round(float(cogs), 2),
        "total_opex": round(float(opex), 2),
        "gross_profit": round(float(gross_profit), 2),
        "net_profit": round(float(net_profit), 2),
        "gross_margin_pct": round(float(gross_margin), 2),
        "net_margin_pct": round(float(net_margin), 2)
    }

def calculate_period_over_period(current_metrics: dict, previous_metrics: dict) -> dict:
    """
    Calculates percentage changes between two periods for key metrics.
    """
    trends = {}
    for key in ['total_revenue', 'gross_profit', 'net_profit']:
        curr = current_metrics.get(key, 0.0)
        prev = previous_metrics.get(key, 0.0)
        if prev > 0:
            pct_change = ((curr - prev) / prev) * 100
        elif curr > 0:
            pct_change = 100.0
        else:
            pct_change = 0.0
        trends[f"{key}_growth_pct"] = round(pct_change, 2)
        
    return trends
