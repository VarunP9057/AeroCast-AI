
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()

RAW_PATH = ROOT / "datasets/Air_Traffic_Passenger_Statistics_20261106.csv"
PROCESSED_DIR = ROOT / "data/processed"
FEATURE_DIR = ROOT / "data/features"
REPORT_DIR = ROOT / "reports/terminal_forecasting"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TERMINALS = [
    "International",
    "Terminal 1",
    "Terminal 2",
    "Terminal 3"
]

TARGET = "Passenger Count"

print("=" * 70)
print("TERMINAL-WISE PASSENGER FEATURE ENGINEERING")
print("=" * 70)

# --------------------------------------------------
# 1. LOAD RAW DATA
# --------------------------------------------------

df = pd.read_csv(RAW_PATH)

print("\nRaw dataset shape:", df.shape)

# --------------------------------------------------
# 2. CLEAN PASSENGER COUNT
# --------------------------------------------------

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df["YearMonth"] = pd.to_datetime(
    df["Activity Period"].astype(str),
    format="%Y%m",
    errors="coerce"
)

df = df.dropna(
    subset=["YearMonth", "Terminal", TARGET]
).copy()

df = df[df["Terminal"].isin(TERMINALS)].copy()

print("Cleaned dataset shape:", df.shape)

# --------------------------------------------------
# 3. AGGREGATE MONTHLY TERMINAL PASSENGER COUNTS
# --------------------------------------------------

monthly = (
    df.groupby(["YearMonth", "Terminal"])[TARGET]
    .sum()
    .reset_index()
)

monthly = monthly.sort_values(
    ["Terminal", "YearMonth"]
).reset_index(drop=True)

print("\nMonthly aggregated records:", len(monthly))

# --------------------------------------------------
# 4. PREPARE CONTINUOUS MONTHLY SERIES
# --------------------------------------------------

terminal_datasets = []

for terminal in TERMINALS:

    terminal_df = monthly[
        monthly["Terminal"] == terminal
    ].copy()

    if terminal == "Terminal 2":
        start_date = pd.Timestamp("2011-04-01")
    else:
        start_date = pd.Timestamp("1999-07-01")

    end_date = pd.Timestamp("2026-08-01")

    expected_months = pd.date_range(
        start=start_date,
        end=end_date,
        freq="MS"
    )

    terminal_df = (
        terminal_df
        .set_index("YearMonth")
        .reindex(expected_months)
    )

    missing_count = terminal_df[TARGET].isna().sum()

    if missing_count > 0:
        raise ValueError(
            f"{terminal} contains {missing_count} missing months. "
            "Investigate before filling missing values."
        )

    terminal_df[TARGET] = terminal_df[TARGET].astype(float)

    terminal_df["Terminal"] = terminal

    terminal_df.index.name = "YearMonth"

    terminal_df = terminal_df.reset_index()

    terminal_datasets.append(terminal_df)

    print(
        f"{terminal}: {len(terminal_df)} records | "
        f"{terminal_df['YearMonth'].min().date()} to "
        f"{terminal_df['YearMonth'].max().date()}"
    )

# --------------------------------------------------
# 5. COMBINE TERMINAL DATASETS
# --------------------------------------------------

combined = pd.concat(
    terminal_datasets,
    ignore_index=True
)

combined = combined.sort_values(
    ["Terminal", "YearMonth"]
).reset_index(drop=True)

combined["Year"] = combined["YearMonth"].dt.year
combined["Month"] = combined["YearMonth"].dt.month

processed_path = (
    PROCESSED_DIR / "terminal_monthly_passengers.csv"
)

combined.to_csv(
    processed_path,
    index=False
)

print("\nProcessed dataset shape:", combined.shape)
print("Saved:", processed_path)

# --------------------------------------------------
# 6. FEATURE ENGINEERING
# --------------------------------------------------

feature_datasets = []

for terminal in TERMINALS:

    terminal_df = combined[
        combined["Terminal"] == terminal
    ].copy()

    terminal_df = terminal_df.sort_values(
        "YearMonth"
    ).reset_index(drop=True)

    # Calendar features

    terminal_df["Quarter"] = (
        terminal_df["YearMonth"].dt.quarter
    )

    # Seasonal features

    terminal_df["Month_Sin"] = np.sin(
        2 * np.pi * terminal_df["Month"] / 12
    )

    terminal_df["Month_Cos"] = np.cos(
        2 * np.pi * terminal_df["Month"] / 12
    )

    # Lag features

    for lag in [1, 2, 3, 6, 12]:
        terminal_df[f"Lag_{lag}"] = (
            terminal_df[TARGET].shift(lag)
        )

    # Rolling mean features

    for window in [3, 6, 12]:
        terminal_df[f"Rolling_Mean_{window}"] = (
            terminal_df[TARGET]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    # Rolling standard deviation features

    for window in [3, 6, 12]:
        terminal_df[f"Rolling_Std_{window}"] = (
            terminal_df[TARGET]
            .shift(1)
            .rolling(window=window)
            .std()
        )

    # Remove rows without sufficient historical data

    terminal_df = terminal_df.dropna().reset_index(drop=True)

    feature_datasets.append(terminal_df)

    print(
        f"\n{terminal} feature records:",
        len(terminal_df)
    )

# --------------------------------------------------
# 7. COMBINE FEATURE DATASETS
# --------------------------------------------------

features = pd.concat(
    feature_datasets,
    ignore_index=True
)

features = features.sort_values(
    ["Terminal", "YearMonth"]
).reset_index(drop=True)

feature_path = (
    FEATURE_DIR / "terminal_monthly_features.csv"
)

features.to_csv(
    feature_path,
    index=False
)

# --------------------------------------------------
# 8. FEATURE COLUMN DEFINITIONS
# --------------------------------------------------

feature_columns = [
    "Year",
    "Month",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_6",
    "Lag_12",
    "Rolling_Mean_3",
    "Rolling_Mean_6",
    "Rolling_Mean_12",
    "Rolling_Std_3",
    "Rolling_Std_6",
    "Rolling_Std_12"
]

# --------------------------------------------------
# 9. SAVE TERMINAL-WISE DATA SUMMARY
# --------------------------------------------------

summary = []

for terminal in TERMINALS:

    terminal_df = features[
        features["Terminal"] == terminal
    ]

    summary.append({
        "Terminal": terminal,
        "Records": len(terminal_df),
        "Start_Date": terminal_df["YearMonth"].min(),
        "End_Date": terminal_df["YearMonth"].max(),
        "Total_Passengers": terminal_df[TARGET].sum(),
        "Average_Monthly_Passengers": terminal_df[TARGET].mean(),
        "Minimum_Monthly_Passengers": terminal_df[TARGET].min(),
        "Maximum_Monthly_Passengers": terminal_df[TARGET].max()
    })

summary_df = pd.DataFrame(summary)

summary_path = (
    REPORT_DIR / "terminal_feature_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

# --------------------------------------------------
# 10. DATA QUALITY CHECKS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print("\nFeature dataset shape:", features.shape)

print("\nFeature columns:")
print(features.columns.tolist())

print("\nTerminal-wise feature summary:")
print(summary_df.to_string(index=False))

print("\nMissing values:")
print(features.isnull().sum().to_string())

print(
    "\nDuplicate terminal-month records:",
    features.duplicated(
        subset=["YearMonth", "Terminal"]
    ).sum()
)

print(
    "\nAvailable model input features:",
    len(feature_columns)
)

print("\nSaved files:")
print(processed_path)
print(feature_path)
print(summary_path)

print("\n" + "=" * 70)
print("TERMINAL FEATURE ENGINEERING COMPLETED")
print("=" * 70)
