from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Date
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_order_id = Column(String, index=True) # order id from raw data
    order_item_id = Column(String, unique=True, index=True)
    customer_id = Column(Integer, index=True)
    customer_segment = Column(String)
    customer_city = Column(String)
    customer_state = Column(String)
    customer_country = Column(String)
    
    order_date = Column(DateTime, index=True)
    order_status = Column(String)
    
    product_category = Column(String)
    product_name = Column(String)
    product_id = Column(Integer, index=True)
    
    quantity = Column(Integer)
    product_price = Column(Float)
    discount = Column(Float)
    discount_rate = Column(Float)
    sales = Column(Float)
    order_item_total = Column(Float)
    order_profit = Column(Float)
    profit_ratio = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Shipment(Base):
    __tablename__ = 'shipments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_item_id = Column(String, ForeignKey('orders.order_item_id'), unique=True, index=True)
    
    shipping_date = Column(DateTime, index=True)
    shipping_mode = Column(String)
    days_for_shipping_real = Column(Integer)
    days_for_shipping_scheduled = Column(Integer)
    delivery_status = Column(String)
    late_delivery_risk = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Inventory(Base):
    __tablename__ = 'inventory'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    product_id = Column(Integer, index=True)
    product_category = Column(String)
    product_name = Column(String)
    
    stock_level = Column(Integer)
    reorder_point = Column(Integer)
    is_stockout = Column(Integer) # 1 or 0
    
    created_at = Column(DateTime, default=datetime.utcnow)

class FinancialTransaction(Base):
    __tablename__ = 'financial_transactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    transaction_type = Column(String, index=True) # 'Revenue', 'COGS', 'OpEx'
    category = Column(String)
    amount = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
