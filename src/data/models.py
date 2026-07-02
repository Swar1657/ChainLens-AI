from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime, date

class RawOrderRecord(BaseModel):
    """Pydantic model for validating raw order records from CSV."""
    model_config = ConfigDict(coerce_numbers_to_str=True)
    
    order_id: int = Field(alias="Order Id")
    order_item_id: int = Field(alias="Order Item Id")
    order_date: str = Field(alias="order date (DateOrders)")
    customer_id: int = Field(alias="Customer Id")
    customer_segment: str = Field(alias="Customer Segment")
    customer_city: str = Field(alias="Customer City")
    customer_state: str = Field(alias="Customer State")
    customer_country: str = Field(alias="Customer Country")
    
    order_status: str = Field(alias="Order Status")
    
    product_category: str = Field(alias="Category Name")
    product_name: str = Field(alias="Product Name")
    product_id: int = Field(alias="Product Card Id")
    
    quantity: int = Field(alias="Order Item Quantity")
    product_price: float = Field(alias="Product Price")
    discount: float = Field(alias="Order Item Discount")
    discount_rate: float = Field(alias="Order Item Discount Rate")
    sales: float = Field(alias="Sales")
    order_item_total: float = Field(alias="Order Item Total")
    order_profit: float = Field(alias="Order Profit Per Order")
    profit_ratio: float = Field(alias="Order Item Profit Ratio")
    
    shipping_date: str = Field(alias="shipping date (DateOrders)")
    shipping_mode: str = Field(alias="Shipping Mode")
    days_for_shipping_real: int = Field(alias="Days for shipping (real)")
    days_for_shipping_scheduled: int = Field(alias="Days for shipment (scheduled)")
    delivery_status: str = Field(alias="Delivery Status")
    late_delivery_risk: int = Field(alias="Late_delivery_risk")

class OrderData(BaseModel):
    source_order_id: str
    order_item_id: str
    customer_id: int
    customer_segment: str
    customer_city: str
    customer_state: str
    customer_country: str
    order_date: datetime
    order_status: str
    product_category: str
    product_name: str
    product_id: int
    quantity: int
    product_price: float
    discount: float
    discount_rate: float
    sales: float
    order_item_total: float
    order_profit: float
    profit_ratio: float

class ShipmentData(BaseModel):
    order_item_id: str
    shipping_date: datetime
    shipping_mode: str
    days_for_shipping_real: int
    days_for_shipping_scheduled: int
    delivery_status: str
    late_delivery_risk: int
