import sys
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.sql.database import DatabaseClient
from src.storage.sql.models import Order, Shipment, Inventory, FinancialTransaction
from src.data.ingestion.loaders import OmniSupplyDataLoader

def main():
    logger.info("Starting Data Ingestion Process")
    
    data_dir = str(Path(__file__).parent.parent / "data" / "raw")
    loader = OmniSupplyDataLoader(data_dir=data_dir)
    
    # 1. Load raw CSV
    try:
        raw_df = loader.load_raw_csv("DataCoSupplyChainDataset.csv")
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        sys.exit(1)
        
    # 2. Process and Validate Pydantic Models
    logger.info("Validating with Pydantic...")
    orders_data, shipments_data = loader.process_and_validate()
    
    # 3. Derive deterministic data
    inv_df = loader.derive_inventory(raw_df)
    fin_df = loader.derive_financials(raw_df)
    
    # 4. Initialize Database
    db_client = DatabaseClient()
    logger.info(f"Connecting to DB: {db_client.database_url}")
    db_client.drop_all() # Reset for clean load during dev
    db_client.init_db()
    
    # 5. Load into Database
    logger.info("Loading into SQL Database...")
    with db_client.get_session() as session:
        # We will insert in batches to avoid memory issues
        # For demonstration purposes, we insert the first 10,000 to keep it manageable in Step 1 testing
        # Or all of it if needed. Let's do 10000 for rapid iteration.
        BATCH_SIZE = 10000
        
        # Load Orders
        logger.info(f"Loading {len(orders_data)} orders...")
        for i in range(len(orders_data)):
            o = orders_data[i]
            db_order = Order(
                source_order_id=o.source_order_id,
                order_item_id=o.order_item_id,
                customer_id=o.customer_id,
                customer_segment=o.customer_segment,
                customer_city=o.customer_city,
                customer_state=o.customer_state,
                customer_country=o.customer_country,
                order_date=o.order_date,
                order_status=o.order_status,
                product_category=o.product_category,
                product_name=o.product_name,
                product_id=o.product_id,
                quantity=o.quantity,
                product_price=o.product_price,
                discount=o.discount,
                discount_rate=o.discount_rate,
                sales=o.sales,
                order_item_total=o.order_item_total,
                order_profit=o.order_profit,
                profit_ratio=o.profit_ratio
            )
            session.add(db_order)
            
            s = shipments_data[i]
            db_shipment = Shipment(
                order_item_id=s.order_item_id,
                shipping_date=s.shipping_date,
                shipping_mode=s.shipping_mode,
                days_for_shipping_real=s.days_for_shipping_real,
                days_for_shipping_scheduled=s.days_for_shipping_scheduled,
                delivery_status=s.delivery_status,
                late_delivery_risk=s.late_delivery_risk
            )
            session.add(db_shipment)
            
            if i % BATCH_SIZE == 0 and i > 0:
                session.commit()
                logger.info(f"Inserted {i} orders/shipments")
        session.commit()
        
        # Load Inventory
        logger.info(f"Loading {len(inv_df)} inventory records...")
        for _, row in inv_df.iterrows():
            db_inv = Inventory(
                date=row['date'],
                product_id=row['product_id'],
                product_category=row['product_category'],
                product_name=row['product_name'],
                stock_level=row['stock_level'],
                reorder_point=row['reorder_point'],
                is_stockout=row['is_stockout']
            )
            session.add(db_inv)
        session.commit()
        
        # Load Financials
        logger.info(f"Loading {len(fin_df)} financial records...")
        for _, row in fin_df.iterrows():
            db_fin = FinancialTransaction(
                date=row['date'],
                transaction_type=row['transaction_type'],
                category=row['category'],
                amount=row['amount']
            )
            session.add(db_fin)
        session.commit()
        
    counts = db_client.get_table_counts()
    logger.info("Data loaded successfully. Database summary:")
    for table, count in counts.items():
        logger.info(f"  - {table}: {count}")
        
    logger.info("Step 1 Data Ingestion Complete.")

if __name__ == "__main__":
    main()
