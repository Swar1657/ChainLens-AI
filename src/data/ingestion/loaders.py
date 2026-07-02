import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dateutil import parser
import logging

from ..models import RawOrderRecord, OrderData, ShipmentData

logger = logging.getLogger(__name__)

class OmniSupplyDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.raw_data = None
        
    def load_raw_csv(self, file_name: str = "DataCoSupplyChainDataset.csv"):
        file_path = f"{self.data_dir}/{file_name}"
        logger.info(f"Loading raw data from {file_path}")
        self.raw_data = pd.read_csv(file_path, encoding='latin1')
        # Clean up column names by stripping spaces just in case
        self.raw_data.columns = self.raw_data.columns.str.strip()
        logger.info(f"Loaded {len(self.raw_data)} rows.")
        return self.raw_data

    def _parse_date(self, date_str: str) -> datetime:
        try:
            return parser.parse(date_str)
        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {e}")
            return datetime.utcnow()

    def process_and_validate(self) -> Tuple[List[OrderData], List[ShipmentData]]:
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw_csv first.")
            
        valid_orders = []
        valid_shipments = []
        
        # We will process a subset for demo/testing or all if production. 
        # Since 180k rows takes time to validate with Pydantic, we will process all of them
        # but optimize by using pandas where possible.
        
        # Let's convert dates in bulk
        df = self.raw_data.copy()
        
        for idx, row in df.iterrows():
            try:
                # Validate with Pydantic
                raw_record = RawOrderRecord(**row.to_dict())
                
                # Normalize
                order_date = self._parse_date(raw_record.order_date)
                shipping_date = self._parse_date(raw_record.shipping_date)
                
                order_item_str = str(raw_record.order_item_id)
                
                order = OrderData(
                    source_order_id=str(raw_record.order_id),
                    order_item_id=order_item_str,
                    customer_id=raw_record.customer_id,
                    customer_segment=raw_record.customer_segment,
                    customer_city=raw_record.customer_city,
                    customer_state=raw_record.customer_state,
                    customer_country=raw_record.customer_country,
                    order_date=order_date,
                    order_status=raw_record.order_status,
                    product_category=raw_record.product_category,
                    product_name=raw_record.product_name,
                    product_id=raw_record.product_id,
                    quantity=raw_record.quantity,
                    product_price=raw_record.product_price,
                    discount=raw_record.discount,
                    discount_rate=raw_record.discount_rate,
                    sales=raw_record.sales,
                    order_item_total=raw_record.order_item_total,
                    order_profit=raw_record.order_profit,
                    profit_ratio=raw_record.profit_ratio
                )
                
                shipment = ShipmentData(
                    order_item_id=order_item_str,
                    shipping_date=shipping_date,
                    shipping_mode=raw_record.shipping_mode,
                    days_for_shipping_real=raw_record.days_for_shipping_real,
                    days_for_shipping_scheduled=raw_record.days_for_shipping_scheduled,
                    delivery_status=raw_record.delivery_status,
                    late_delivery_risk=raw_record.late_delivery_risk
                )
                
                valid_orders.append(order)
                valid_shipments.append(shipment)
                
            except Exception as e:
                # Log first few errors
                if len(valid_orders) < 5:
                    logger.warning(f"Validation error at row {idx}: {e}")
                continue
                
        logger.info(f"Validated {len(valid_orders)} orders and shipments.")
        return valid_orders, valid_shipments

    def derive_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives inventory levels deterministically based on real orders.
        Initial stock is assigned per product. Stock depletes over time with each order.
        When stock drops below reorder point, it is replenished.
        """
        logger.info("Deriving synthetic inventory from real order constraints...")
        
        # Sort orders chronologically
        df['order_date_parsed'] = pd.to_datetime(df['order date (DateOrders)'], format='mixed', errors='coerce')
        sorted_df = df.dropna(subset=['order_date_parsed']).sort_values('order_date_parsed')
        
        inventory_records = []
        product_stock = {}
        
        # Simplify by processing daily aggregations
        daily_sales = sorted_df.groupby([sorted_df['order_date_parsed'].dt.date, 'Product Card Id', 'Category Name', 'Product Name'])['Order Item Quantity'].sum().reset_index()
        
        for _, row in daily_sales.iterrows():
            d = row['order_date_parsed']
            pid = row['Product Card Id']
            cat = row['Category Name']
            name = row['Product Name']
            qty = int(row['Order Item Quantity'])
            
            if pid not in product_stock:
                # Initialize stock based on first day demand to simulate realistic levels
                product_stock[pid] = max(100, qty * 10)
                
            reorder_point = 20
            
            # Deplete
            product_stock[pid] -= qty
            is_stockout = 1 if product_stock[pid] <= 0 else 0
            
            inventory_records.append({
                "date": d,
                "product_id": pid,
                "product_category": cat,
                "product_name": name,
                "stock_level": product_stock[pid],
                "reorder_point": reorder_point,
                "is_stockout": is_stockout
            })
            
            # Restock logic (happens next day)
            if product_stock[pid] <= reorder_point:
                product_stock[pid] += 100 # fixed restock amount
                
        inv_df = pd.DataFrame(inventory_records)
        logger.info(f"Generated {len(inv_df)} daily inventory records.")
        return inv_df
        
    def derive_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives financial transactions deterministically.
        Extracts revenue and COGS directly from orders.
        Generates OpEx as a fixed percentage of daily revenue (only because architecture requires non-order financials).
        """
        logger.info("Deriving financial transactions...")
        df['order_date_parsed'] = pd.to_datetime(df['order date (DateOrders)'], format='mixed', errors='coerce')
        
        # Daily Revenue & COGS
        daily_fin = df.groupby(df['order_date_parsed'].dt.date).agg(
            Revenue=('Sales', 'sum'),
            Profit=('Order Profit Per Order', 'sum')
        ).reset_index()
        
        fin_records = []
        for _, row in daily_fin.iterrows():
            d = row['order_date_parsed']
            rev = row['Revenue']
            profit = row['Profit']
            cogs = rev - profit
            
            # Real derived values
            fin_records.append({"date": d, "transaction_type": "Revenue", "category": "Sales", "amount": rev})
            fin_records.append({"date": d, "transaction_type": "COGS", "category": "Cost of Goods", "amount": cogs})
            
            # Synthetic OpEx (fixed 15% of revenue) to fulfill architecture
            opex = rev * 0.15
            fin_records.append({"date": d, "transaction_type": "OpEx", "category": "Operating Expenses", "amount": opex})
            
        fin_df = pd.DataFrame(fin_records)
        logger.info(f"Generated {len(fin_df)} financial transaction records.")
        return fin_df
