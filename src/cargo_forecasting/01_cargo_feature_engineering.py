
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/cargo_monthly.csv"
OUTPUT_DIR = ROOT / "data/features"
OUTPUT_FILE = OUTPUT_DIR / "cargo_monthly_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print("\nOriginal Dataset Shape:", df.shape)
print("\nOriginal Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 2. DATA VALIDATION
# --------------------------------------------------

required_columns = [
    "YearMonth",
    "Cargo Weight LBS",
    "Year",
    "Month"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

if df["YearMonth"].duplicated().any():
    raise ValueError("Duplicate months found in cargo dataset.")

if df["Cargo Weight LBS"].isna().any():
    raise ValueError("Missing cargo target values found.")

if (df["Cargo Weight LBS"] < 0).any():
    raise ValueError("Negative cargo weight values found.")


# --------------------------------------------------
# 3. CALENDAR FEATURES
# --------------------------------------------------

df["Year"] = df["YearMonth"].dt.year
df["Month"] = df["YearMonth"].dt.month

df["Quarter"] = df["YearMonth"].dt.quarter


# --------------------------------------------------
# 4. SEASONALITY FEATURES
# --------------------------------------------------

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)


# --------------------------------------------------
# 5. LAG FEATURES
# --------------------------------------------------

for lag in [1, 2, 3, 6, 12]:
    df[f"Lag_{lag}"] = df["Cargo Weight LBS"].shift(lag)


# --------------------------------------------------
# 6. ROLLING MEAN FEATURES
# --------------------------------------------------

for window in [3, 6]:
    df[f"Rolling_Mean_{window}"] = (
        df["Cargo Weight LBS"]
        .shift(1)
        .rolling(window=window)
        .mean()
    )


# --------------------------------------------------
# 7. ROLLING STANDARD DEVIATION
# --------------------------------------------------

for window in [3, 6]:
    df[f"Rolling_Std_{window}"] = (
        df["Cargo Weight LBS"]
        .shift(1)
        .rolling(window=window)
        .std()
    )


# --------------------------------------------------
# 8. REMOVE ROWS WITH INSUFFICIENT HISTORY
# --------------------------------------------------

df = df.dropna().reset_index(drop=True)


# --------------------------------------------------
# 9. FINAL DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("CARGO FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nFinal Dataset Shape:", df.shape)

print("\nFeatures:")
print(df.columns.tolist())

print("\nDate Range:")
print(df["YearMonth"].min(), "to", df["YearMonth"].max())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst Five Records:")
print(df.head())

print("\nSaved Dataset:")
print(OUTPUT_FILE)
