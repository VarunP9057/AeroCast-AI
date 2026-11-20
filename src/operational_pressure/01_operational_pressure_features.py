
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()

INPUT_PATH = ROOT / "data/processed/integrated_monthly_airport_data.csv"
OUTPUT_DIR = ROOT / "data/features"
REPORT_DIR = ROOT / "reports/operational_pressure"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "operational_pressure_features.csv"

df = pd.read_csv(INPUT_PATH)

df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

df["Year"] = df["YearMonth"].dt.year
df["Month"] = df["YearMonth"].dt.month
df["Quarter"] = df["YearMonth"].dt.quarter

activity_columns = [
    "Passenger Count",
    "Landing Count",
    "Cargo Weight LBS"
]

if df[activity_columns].isnull().any().any():
    raise ValueError("Missing activity values found.")

if (df[activity_columns] < 0).any().any():
    raise ValueError("Negative activity values found.")

# Define the chronological training period
train_mask = df["YearMonth"] < "2021-06-01"

if train_mask.sum() == 0:
    raise ValueError("Training period is empty.")

# Normalize using training-period statistics only
for column, name in [
    ("Passenger Count", "Passenger"),
    ("Landing Count", "Aircraft"),
    ("Cargo Weight LBS", "Cargo")
]:
    mean_value = df.loc[train_mask, column].mean()
    std_value = df.loc[train_mask, column].std()

    if std_value == 0:
        raise ValueError(f"Zero standard deviation for {column}")

    df[f"{name}_Normalized"] = (
        (df[column] - mean_value) / std_value
    )

# Convert normalized activity to a 0-100 index
# Equal weights are an initial transparent assumption
combined_score = (
    df["Passenger_Normalized"]
    + df["Aircraft_Normalized"]
    + df["Cargo_Normalized"]
) / 3

train_min = combined_score[train_mask].min()
train_max = combined_score[train_mask].max()

if train_max == train_min:
    raise ValueError("Cannot scale a constant pressure score.")

df["Operational Pressure Index"] = (
    (combined_score - train_min)
    / (train_max - train_min)
) * 100

# Calendar seasonality
df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

# Time trend
df["Time_Index"] = np.arange(len(df))

TARGET = "Operational Pressure Index"

# Past-only lag and rolling features
for lag in [1, 2, 3, 6, 12]:
    df[f"Pressure_Lag_{lag}"] = df[TARGET].shift(lag)

for window in [3, 6, 12]:
    df[f"Pressure_Rolling_Mean_{window}"] = (
        df[TARGET].shift(1).rolling(window).mean()
    )

    df[f"Pressure_Rolling_Std_{window}"] = (
        df[TARGET].shift(1).rolling(window).std()
    )

# Activity-specific lag features
for column, name in [
    ("Passenger Count", "Passenger"),
    ("Landing Count", "Aircraft"),
    ("Cargo Weight LBS", "Cargo")
]:
    df[f"{name}_Lag_1"] = df[column].shift(1)
    df[f"{name}_Lag_12"] = df[column].shift(12)

# Remove rows without sufficient historical features
feature_df = df.dropna().reset_index(drop=True)

# Save the complete engineered dataset
feature_df.to_csv(OUTPUT_PATH, index=False)

# Save index normalization information for future use
normalization = pd.DataFrame({
    "Indicator": ["Passenger", "Aircraft", "Cargo"],
    "Training_Mean": [
        df.loc[train_mask, "Passenger Count"].mean(),
        df.loc[train_mask, "Landing Count"].mean(),
        df.loc[train_mask, "Cargo Weight LBS"].mean()
    ],
    "Training_Std": [
        df.loc[train_mask, "Passenger Count"].std(),
        df.loc[train_mask, "Landing Count"].std(),
        df.loc[train_mask, "Cargo Weight LBS"].std()
    ]
})

normalization.to_csv(
    REPORT_DIR / "operational_pressure_normalization.csv",
    index=False
)

index_scaling = pd.DataFrame({
    "Training_Score_Min": [train_min],
    "Training_Score_Max": [train_max],
    "Weights": ["Passenger=1/3, Aircraft=1/3, Cargo=1/3"]
})

index_scaling.to_csv(
    REPORT_DIR / "operational_pressure_index_scaling.csv",
    index=False
)

print("=" * 60)
print("OPERATIONAL PRESSURE FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nOriginal dataset:", df.shape)
print("Engineered dataset:", feature_df.shape)

print("\nDate range:")
print(feature_df["YearMonth"].min(), "to", feature_df["YearMonth"].max())

print("\nTraining records:", (feature_df["YearMonth"] < "2021-06-01").sum())
print("Testing records:", (feature_df["YearMonth"] >= "2021-06-01").sum())

print("\nOperational Pressure Index statistics:")
print(feature_df[TARGET].describe().round(3))

print("\nMissing values:")
print(feature_df.isnull().sum().sum())

print("\nSaved dataset:", OUTPUT_PATH)
print("Saved reports:", REPORT_DIR)
