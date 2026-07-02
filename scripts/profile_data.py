import pandas as pd
import json
from pathlib import Path
import sys

def profile_data(csv_path: str, output_path: str):
    print(f"Loading data from {csv_path}...")
    try:
        # Some columns in DataCo dataset might have mixed types or encoding issues. 
        # using latin1 is common for DataCo.
        df = pd.read_csv(csv_path, encoding='latin1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
        
    print(f"Successfully loaded {len(df)} rows.")
    
    profile = {
        "file_name": Path(csv_path).name,
        "row_count": len(df),
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }
    
    # Financial fields check
    financial_fields = [col for col in df.columns if any(term in col.lower() for term in ['sales', 'profit', 'discount', 'price', 'revenue', 'cost'])]
    profile['financial_fields'] = financial_fields
    
    # Delivery fields check
    delivery_fields = [col for col in df.columns if any(term in col.lower() for term in ['shipping', 'delivery', 'status', 'date'])]
    profile['delivery_fields'] = delivery_fields
    
    # Check for date columns and get range
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    date_ranges = {}
    for col in date_columns:
        try:
            temp_date = pd.to_datetime(df[col], errors='coerce')
            valid_dates = temp_date.dropna()
            if len(valid_dates) > 0:
                date_ranges[col] = {
                    "min": valid_dates.min().isoformat(),
                    "max": valid_dates.max().isoformat()
                }
        except:
            pass
    profile['date_ranges'] = date_ranges
    
    # Check categorical cardinalities for string columns
    categorical_cardinalities = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col not in date_columns:  # skip dates
            nunique = df[col].nunique()
            if nunique < 1000: # only log reasonable categorical sizes
                categorical_cardinalities[col] = nunique
    profile['categorical_cardinalities'] = categorical_cardinalities
    
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=4)
        
    print(f"Profile saved to {output_path}")

if __name__ == "__main__":
    csv_file = "d:/projects/ChainLens AI/data/raw/DataCoSupplyChainDataset.csv"
    output_file = "d:/projects/ChainLens AI/data/raw/data_profile.json"
    profile_data(csv_file, output_file)
