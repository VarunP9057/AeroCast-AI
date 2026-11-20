
import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

dataset_folder = project_path / "datasets"

files = list(dataset_folder.glob("Air_Traffic_Cargo_Statistics*.csv"))

if not files:
    raise FileNotFoundError("Cargo dataset not found in datasets folder.")

file_path = files[0]

df = pd.read_csv(file_path, low_memory=False)

print("\n========== AEROCAST AI ==========")
print("CARGO DATASET INSPECTION")
print("================================")

print("\n1. Dataset File")
print(file_path.name)

print("\n2. Dataset Shape")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\n3. Column Names")
for column in df.columns:
    print(column)

print("\n4. First 5 Records")
print(df.head())

print("\n5. Dataset Information")
df.info()

print("\n6. Missing Values")
print(df.isnull().sum())

print("\n7. Missing Value Percentage")
print((df.isnull().mean() * 100).round(2))

print("\n8. Duplicate Records")
print("Total Duplicates:", df.duplicated().sum())

print("\n9. Statistical Summary")
print(df.describe(include="all").T)

print("\n10. Date and Period Columns")

for column in df.columns:
    if "date" in column.lower() or "period" in column.lower():
        print("\nColumn:", column)
        print(df[column].head(10).tolist())

print("\n========== INSPECTION COMPLETED ==========")
