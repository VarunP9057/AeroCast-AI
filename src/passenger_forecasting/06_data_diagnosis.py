
import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

raw_path = project_path / "data" / "processed" / "passenger_cleaned.csv"
feature_path = project_path / "data" / "features" / "passenger_features.csv"

raw = pd.read_csv(raw_path)
features = pd.read_csv(feature_path)

date_col = "Activity Period Start Date"
target = "Passenger Count"

raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
raw[target] = pd.to_numeric(raw[target], errors="coerce")

raw = raw.dropna(subset=[date_col, target])
raw["YearMonth"] = raw[date_col].dt.to_period("M")

monthly = raw.groupby("YearMonth")[target].sum()

print("\n========== RAW DATA ==========")
print("Dataset shape:", raw.shape)
print("Date range:", raw[date_col].min(), "to", raw[date_col].max())
print("\nPassenger statistics:")
print(raw[target].describe())
print("\nTotal passenger count:", raw[target].sum())

print("\n========== MONTHLY AGGREGATION ==========")
print("Number of months:", len(monthly))
print("Average monthly passengers:", monthly.mean())
print("Minimum monthly passengers:", monthly.min())
print("Maximum monthly passengers:", monthly.max())

print("\nFirst 10 monthly totals:")
print(monthly.head(10))

print("\nLast 10 monthly totals:")
print(monthly.tail(10))

print("\n========== FEATURE DATA ==========")
print("Feature dataset shape:", features.shape)
print("\nFirst 10 feature rows:")
print(features.head(10).to_string(index=False))

print("\n========== LAG CHECK ==========")
first_month = features.iloc[0]["YearMonth"]
first_lag = features.iloc[0]["Lag_1"]

print("First feature month:", first_month)
print("Its Lag_1 value:", first_lag)
