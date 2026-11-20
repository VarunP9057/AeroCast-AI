import pandas as pd
import numpy as np
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

input_path = (
    project_path / "data" / "processed"
    / "passenger_cleaned.csv"
)

output_folder = project_path / "data" / "features"
output_folder.mkdir(parents=True, exist_ok=True)

if not input_path.exists():
    raise FileNotFoundError(
        "Cleaned passenger dataset not found. Run 02_clean_data.py first."
    )

df = pd.read_csv(input_path)

print("\n========== AEROCAST AI ==========")
print("PASSENGER FEATURE ENGINEERING")
print("================================")

date_col = "Activity Period Start Date"
target_col = "Passenger Count"

if date_col not in df.columns or target_col not in df.columns:
    raise ValueError("Required passenger date or target column is missing.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

df[target_col] = pd.to_numeric(
    df[target_col], errors="coerce"
)

df = df.dropna(subset=[date_col, target_col])

# 1. Aggregate passenger traffic by month

df["YearMonth"] = df[date_col].dt.to_period("M")

monthly_data = (
    df.groupby("YearMonth")[target_col]
    .sum()
    .sort_index()
    .to_frame()
)

# Include missing calendar months as zero activity

full_months = pd.period_range(
    start=monthly_data.index.min(),
    end=monthly_data.index.max(),
    freq="M"
)

monthly_data = monthly_data.reindex(full_months, fill_value=0)

monthly_data.index.name = "YearMonth"

monthly_data["Year"] = monthly_data.index.year
monthly_data["Month"] = monthly_data.index.month
monthly_data["Quarter"] = monthly_data.index.quarter

# 2. Create seasonal features

monthly_data["Month_Sin"] = np.sin(
    2 * np.pi * monthly_data["Month"] / 12
)

monthly_data["Month_Cos"] = np.cos(
    2 * np.pi * monthly_data["Month"] / 12
)

# 3. Create previous-month passenger features

for lag in [1, 2, 3, 6, 12]:
    monthly_data[f"Lag_{lag}"] = (
        monthly_data[target_col].shift(lag)
    )

# 4. Create rolling averages using previous months only

monthly_data["Rolling_Mean_3"] = (
    monthly_data[target_col]
    .shift(1)
    .rolling(window=3)
    .mean()
)

monthly_data["Rolling_Mean_6"] = (
    monthly_data[target_col]
    .shift(1)
    .rolling(window=6)
    .mean()
)

# 5. Save monthly passenger dataset

monthly_output = monthly_data.reset_index()

monthly_output["YearMonth"] = (
    monthly_output["YearMonth"].astype(str)
)

monthly_output.to_csv(
    output_folder / "passenger_monthly_features_all.csv",
    index=False
)

# 6. Remove rows with missing lag features

features_df = monthly_output.dropna().copy()

# 7. Save final machine learning dataset

features_df.to_csv(
    output_folder / "passenger_features.csv",
    index=False
)

# 8. Display results

print("\n1. Monthly Dataset Shape")
print(monthly_output.shape)

print("\n2. Feature Dataset Shape")
print(features_df.shape)

print("\n3. Feature Columns")
print(features_df.columns.tolist())

print("\n4. First 10 Records")
print(features_df.head(10).to_string(index=False))

print("\n5. Missing Values")
print(features_df.isnull().sum())

print("\n6. Target Statistics")
print(features_df[target_col].describe())

print("\n7. Generated Files")
print("passenger_monthly_features_all.csv")
print("passenger_features.csv")

print("\nSaved in:", output_folder)

print("\n========== FEATURE ENGINEERING COMPLETED ==========")
