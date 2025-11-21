import kagglehub
import pandas as pd
import os
import shutil

print("Downloading NBA games dataset...")
path = kagglehub.dataset_download("nathanlauga/nba-games")
print(f"Dataset downloaded to: {path}")

# Create data directory in your project
os.makedirs("data/raw", exist_ok=True)

# Copy files to your project
print("\nCopying files to data/raw/...")
for file in os.listdir(path):
    if file.endswith('.csv'):
        shutil.copy(
            os.path.join(path, file),
            os.path.join("data/raw", file)
        )
        print(f"✓ {file}")

# Quick data check
print("\n" + "="*50)
print("DATA SUMMARY")
print("="*50)

for file in os.listdir("data/raw"):
    if file.endswith('.csv'):
        df = pd.read_csv(f"data/raw/{file}")
        print(f"\n{file}:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        
print("\n✓ Setup complete! Data is ready in data/raw/")