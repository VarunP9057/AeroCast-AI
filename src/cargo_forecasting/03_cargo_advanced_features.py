 
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/cargo_monthly.csv"
OUTPUT_DIR = ROOT / "data/features"
OUTPUT_FILE = OUTPUT_DIR / "cargo_advanced_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

target = "Cargo Weight LBS"

if df["YearMonth"].duplicated().any():
    raise ValueError("Duplicate months found.")

if df[target].isna().any():
    raise ValueError("Missing cargo target values found.")

if (df[target] < 0).any():
    raise ValueError("Negative cargo values found.")


# --------------------------------------------------
# 2. CALENDAR FEATURES
# --------------------------------------------------

df["Year"] = df["YearMonth"].dt.year
df["Month"] = df["YearMonth"].dt.month
df["Quarter"] = df["YearMonth"].dt.quarter

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

df["Time_Index"] = np.arange(len(df))


# --------------------------------------------------
# 3. EXTENDED LAG FEATURES
# --------------------------------------------------

for lag in [1, 2, 3, 6, 12, 18, 24]:
    df[f"Lag_{lag}"] = df[target].shift(lag)


# --------------------------------------------------
# 4. EXTENDED ROLLING FEATURES
# --------------------------------------------------

shifted_target = df[target].shift(1)

for window in [3, 6, 12, 24]:
    df[f"Rolling_Mean_{window}"] = (
        shifted_target.rolling(window=window).mean()
    )

for window in [3, 6, 12]:
    df[f"Rolling_Std_{window}"] = (
        shifted_target.rolling(window=window).std()
    )


# --------------------------------------------------
# 5. YEAR-OVER-YEAR CHANGE
# --------------------------------------------------

df["Previous_Year_Change"] = (
    df[target].shift(1) - df[target].shift(13)
)

df["Previous_Year_Change_Pct"] = (
    df["Previous_Year_Change"] /
    df[target].shift(13).replace(0, np.nan)
) * 100


# --------------------------------------------------
# 6. RECENT TREND FEATURES
# --------------------------------------------------

df["Trend_Slope_6"] = (
    shifted_target
    .rolling(6)
    .apply(
        lambda values: np.polyfit(
            np.arange(len(values)),
            values,
            1
        )[0],
        raw=True
    )
)

df["Trend_Slope_12"] = (
    shifted_target
    .rolling(12)
    .apply(
        lambda values: np.polyfit(
            np.arange(len(values)),
            values,
            1
        )[0],
        raw=True
    )
)


# --------------------------------------------------
# 7. PAST-ONLY MONTHLY SEASONAL AVERAGE
# --------------------------------------------------

df["Historical_Month_Average"] = (
    df.groupby("Month")[target]
    .transform(
        lambda values: values.shift(1).expanding().mean()
    )
)


# --------------------------------------------------
# 8. REMOVE INSUFFICIENT HISTORY
# --------------------------------------------------

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)


# --------------------------------------------------
# 9. VALIDATE FINAL DATASET
# --------------------------------------------------

if df.isna().any().any():
    raise ValueError("Missing values remain in advanced features.")

if df["YearMonth"].duplicated().any():
    raise ValueError("Duplicate months found in final dataset.")


# --------------------------------------------------
# 10. SAVE DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("CARGO ADVANCED FEATURE ENGINEERING COMPLETED")
print("=" * 65)

print("\nDataset Shape:", df.shape)

print("\nDate Range:")
print(df["YearMonth"].min(), "to", df["YearMonth"].max())

print("\nTotal Features:", len(df.columns) - 2)

print("\nFeature Columns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nFirst Five Records:")
print(df.head())

print("\nSaved Dataset:")
print(OUTPUT_FILE)
