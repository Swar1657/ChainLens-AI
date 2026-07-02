import pytest
from datetime import datetime
import pandas as pd

from src.data.models import OrderData, ShipmentData
from src.data.ingestion.loaders import OmniSupplyDataLoader

def test_order_model_validation():
    order = OrderData(
        source_order_id="12345",
        order_item_id="987",
        customer_id=1,
        customer_segment="Consumer",
        customer_city="New York",
        customer_state="NY",
        customer_country="USA",
        order_date=datetime(2023, 1, 1),
        order_status="COMPLETE",
        product_category="Electronics",
        product_name="Phone",
        product_id=101,
        quantity=2,
        product_price=500.0,
        discount=10.0,
        discount_rate=0.01,
        sales=1000.0,
        order_item_total=990.0,
        order_profit=200.0,
        profit_ratio=0.2
    )
    assert order.source_order_id == "12345"
    assert order.quantity == 2

def test_derive_financials():
    loader = OmniSupplyDataLoader(data_dir="dummy")
    
    # Mock some basic dataframe rows
    df = pd.DataFrame([
        {
            "order date (DateOrders)": "2023-01-01 10:00:00",
            "Sales": 1000.0,
            "Order Profit Per Order": 200.0
        },
        {
            "order date (DateOrders)": "2023-01-01 14:00:00",
            "Sales": 500.0,
            "Order Profit Per Order": 50.0
        }
    ])
    
    fin_df = loader.derive_financials(df)
    
    # Should have 3 records for 2023-01-01: Revenue, COGS, OpEx
    assert len(fin_df) == 3
    
    revenue_row = fin_df[fin_df['transaction_type'] == 'Revenue'].iloc[0]
    assert revenue_row['amount'] == 1500.0
    
    cogs_row = fin_df[fin_df['transaction_type'] == 'COGS'].iloc[0]
    assert cogs_row['amount'] == 1250.0 # 1500 - 250
    
    opex_row = fin_df[fin_df['transaction_type'] == 'OpEx'].iloc[0]
    assert opex_row['amount'] == 1500.0 * 0.15
