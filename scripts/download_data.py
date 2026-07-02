import os
import sys
import zipfile
from pathlib import Path
import subprocess

def check_kaggle_auth():
    """Check if Kaggle credentials exist."""
    kaggle_dir = Path.home() / ".kaggle"
    return (kaggle_dir / "kaggle.json").exists()

def download_dataco_dataset(data_dir: Path):
    """Download the DataCo dataset from Kaggle."""
    dataset_name = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
    
    print(f"Downloading dataset {dataset_name}...")
    try:
        # Run kaggle command
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(data_dir)],
            check=True
        )
        
        # Unzip
        zip_path = data_dir / "dataco-smart-supply-chain-for-big-data-analysis.zip"
        if zip_path.exists():
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            # Remove zip
            zip_path.unlink()
            print("Download and extraction complete.")
        else:
            print(f"Error: Could not find downloaded zip file at {zip_path}")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset. Ensure Kaggle CLI is installed and authenticated. Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it using 'pip install kaggle'.")
        sys.exit(1)

def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not check_kaggle_auth():
        print("Error: Kaggle authentication not found!")
        print("To download the dataset, you need a Kaggle account.")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' to download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")
        print(f"And place the CSV files in: {data_dir.absolute()}")
        sys.exit(1)
        
    download_dataco_dataset(data_dir)

if __name__ == "__main__":
    main()
