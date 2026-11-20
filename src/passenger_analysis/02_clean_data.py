import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

input_file = project_path / "datasets" / "Air_Traffic_Passenger_Statistics_20261106.csv"

output_folder = project_path / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "passenger_cleaned.csv"

df = pd.read_csv(input_file, low_memory=False)

print("\n========== PASSENGER DATA CLEANING ==========")

print("\n1. Original Dataset Shape")
print(df.shape)

print("\n2. Missing Values Before Cleaning")
print(df.isnull().sum())

print("\n3. Removing Exact Duplicate Records")
duplicates = df.duplicated().sum()
print("Duplicates Found:", duplicates)

df = df.drop_duplicates()

print("\n4. Cleaning Column Names")
df.columns = df.columns.str.strip()

print("\n5. Cleaning Text Columns")
text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = df[column].str.strip()
    df[column] = df[column].replace("", pd.NA)

print("\n6. Converting Date Column")
df["Activity Period Start Date"] = pd.to_datetime(
    df["Activity Period Start Date"],
    errors="coerce"
)

print("\n7. Correcting Passenger Count")
df["Passenger Count"] = (
    df["Passenger Count"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Passenger Count"] = pd.to_numeric(
    df["Passenger Count"],
    errors="coerce"
)

print("\n8. Removing Rows With Missing Date or Passenger Count")
before = len(df)

df = df.dropna(
    subset=["Activity Period Start Date", "Passenger Count"]
)

print("Rows Removed:", before - len(df))

print("\n9. Filling Missing Categorical Values")
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

print("\n10. Final Dataset Shape")
print(df.shape)

print("\n11. Passenger Count Statistics")
print(df["Passenger Count"].describe())

print("\nTotal Passenger Count:", df["Passenger Count"].sum())

print("\n12. Missing Values After Cleaning")
print(df.isnull().sum())

print("\n13. Saving Cleaned Dataset")
df.to_csv(output_file, index=False)

print("Saved:", output_file)

print("\n14. Cleaned Dataset Preview")
print(df.head())

print("\n========== CLEANING COMPLETED ==========")
