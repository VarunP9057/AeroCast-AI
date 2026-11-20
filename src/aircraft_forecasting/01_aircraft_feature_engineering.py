
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aircraft_monthly.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "features"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "aircraft_monthly_features.csv"

df = pd.read_csv(INPUT_FILE)

df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m")

df = df.sort_values("YearMonth").reset_index(drop=True)

df["Landing Count"] = pd.to_numeric(
    df["Landing Count"], errors="coerce"
)

df["Total Landed Weight"] = pd.to_numeric(
    df["Total Landed Weight"], errors="coerce"
)

if df[["YearMonth", "Landing Count"]].isnull().any().any():
    raise ValueError("Missing dates or landing counts found.")

if df["YearMonth"].duplicated().any():
    raise ValueError("Duplicate monthly records found.")

expected_months = pd.date_range(
    start=df["YearMonth"].min(),
    end=df["YearMonth"].max(),
    freq="MS"
)

if not df["YearMonth"].reset_index(drop=True).equals(
    pd.Series(expected_months, name="YearMonth")
):
    raise ValueError("Missing or irregular calendar months detected.")

df["Year"] = df["YearMonth"].dt.year
df["Month"] = df["YearMonth"].dt.month
df["Quarter"] = df["YearMonth"].dt.quarter

df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

for lag in [1, 2, 3, 6, 12]:
    df[f"Lag_{lag}"] = df["Landing Count"].shift(lag)

df["Rolling_Mean_3"] = (
    df["Landing Count"].shift(1).rolling(window=3).mean()
)

df["Rolling_Mean_6"] = (
    df["Landing Count"].shift(1).rolling(window=6).mean()
)

df["Rolling_Std_3"] = (
    df["Landing Count"].shift(1).rolling(window=3).std()
)

df["Rolling_Std_6"] = (
    df["Landing Count"].shift(1).rolling(window=6).std()
)

df = df.dropna().reset_index(drop=True)

df["YearMonth"] = df["YearMonth"].dt.strftime("%Y-%m")

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== AEROCAST AI ==========")
print("AIRCRAFT FEATURE ENGINEERING")
print("================================")

print("\nDataset shape:", df.shape)
print("\nFeatures:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDate range:", df["YearMonth"].min(), "to", df["YearMonth"].max())

print("\nFeature dataset saved to:")
print(OUTPUT_FILE)

print("\n========== FEATURE ENGINEERING COMPLETED ==========")
