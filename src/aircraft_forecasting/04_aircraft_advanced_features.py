
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "aircraft_monthly.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "features"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "aircraft_advanced_features.csv"

df = pd.read_csv(INPUT_FILE)

df["YearMonth"] = pd.to_datetime(
    df["YearMonth"], format="%Y-%m"
)

df = df.sort_values("YearMonth").reset_index(drop=True)

df["Landing Count"] = pd.to_numeric(
    df["Landing Count"], errors="coerce"
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

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

# Historical lag features

for lag in [1, 2, 3, 6, 9, 12, 18, 24]:
    df[f"Lag_{lag}"] = df["Landing Count"].shift(lag)

# Year-over-year features

df["YoY_Change"] = (
    df["Landing Count"].shift(12)
    - df["Landing Count"].shift(24)
)

df["YoY_Growth_Percent"] = (
    (
        df["Landing Count"].shift(12)
        - df["Landing Count"].shift(24)
    )
    / df["Landing Count"].shift(24).replace(0, np.nan)
) * 100

# Rolling mean features

for window in [3, 6, 9, 12]:
    df[f"Rolling_Mean_{window}"] = (
        df["Landing Count"]
        .shift(1)
        .rolling(window=window)
        .mean()
    )

# Rolling standard deviation

for window in [3, 6, 12]:
    df[f"Rolling_Std_{window}"] = (
        df["Landing Count"]
        .shift(1)
        .rolling(window=window)
        .std()
    )

# Recent traffic momentum

df["Recent_Trend"] = (
    df["Landing Count"].shift(1)
    - df["Landing Count"].shift(4)
)

# Seasonal historical average

df["Seasonal_Lag_Average"] = (
    df["Landing Count"]
    .shift(1)
    .rolling(window=12)
    .mean()
)

df = df.dropna().reset_index(drop=True)

df["YearMonth"] = df["YearMonth"].dt.strftime("%Y-%m")

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== AEROCAST AI ==========")
print("ADVANCED AIRCRAFT FEATURE ENGINEERING")
print("================================")

print("\nDataset Shape:", df.shape)

print("\nFeature Columns:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head().to_string(index=False))

print("\nLast 5 Records:")
print(df.tail().to_string(index=False))

print("\nMissing Values:")
print(df.isnull().sum().to_string())

print("\nDate Range:")
print(df["YearMonth"].min(), "to", df["YearMonth"].max())

print("\nSaved Dataset:")
print(OUTPUT_FILE)

print("\n========== FEATURE ENGINEERING COMPLETED ==========")
