
import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

input_path = (
    project_path / "data" / "processed"
    / "integrated_monthly_airport_data.csv"
)

output_folder = project_path / "reports" / "evaluation"
output_folder.mkdir(parents=True, exist_ok=True)

if not input_path.exists():
    raise FileNotFoundError(
        "Integrated dataset not found. Run 01_monthly_aggregation.py first."
    )

df = pd.read_csv(input_path)

required_columns = [
    "YearMonth",
    "Passenger Count",
    "Landing Count",
    "Cargo Weight LBS",
    "Total Landed Weight",
    "Year",
    "Month"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["YearMonth"] = pd.to_datetime(
    df["YearMonth"], errors="coerce"
)

metrics = [
    "Passenger Count",
    "Landing Count",
    "Cargo Weight LBS"
]

for col in metrics + ["Total Landed Weight"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=["YearMonth"] + metrics)
    .sort_values("YearMonth")
    .reset_index(drop=True)
)

print("\n========== AEROCAST AI ==========")
print("INTEGRATED OPERATIONAL ANALYSIS")
print("================================")

print("\n1. Dataset Overview")
print("Total Months:", len(df))
print("Date Range:", df["YearMonth"].min(), "to", df["YearMonth"].max())
print("Metrics:", metrics)

print("\n2. Overall Airport Performance")

summary = df[metrics + ["Total Landed Weight"]].describe().T

summary.to_csv(
    output_folder / "integrated_statistical_summary.csv"
)

print(summary.round(2).to_string())

print("\n3. Yearly Airport Performance")

yearly = (
    df.groupby("Year")[metrics + ["Total Landed Weight"]]
    .sum(min_count=1)
    .reset_index()
)

yearly.to_csv(
    output_folder / "integrated_yearly_performance.csv",
    index=False
)

print(yearly.tail(10).to_string(index=False))

print("\n4. Relative Operational Pressure Index")

# Normalize each activity metric to a 0-100 scale.
# This is a historical relative-activity measure,
# not a direct measurement of physical airport congestion.

for col in metrics:
    minimum = df[col].min()
    maximum = df[col].max()

    if maximum == minimum:
        df[col + " Normalized"] = 0
    else:
        df[col + " Normalized"] = (
            (df[col] - minimum) / (maximum - minimum)
        ) * 100

# Proposed analytical weights:
# Passenger activity: 40%
# Aircraft landing activity: 40%
# Cargo activity: 20%

df["Operational Pressure Index"] = (
    df["Passenger Count Normalized"] * 0.40
    + df["Landing Count Normalized"] * 0.40
    + df["Cargo Weight LBS Normalized"] * 0.20
)

df["Operational Pressure Index"] = (
    df["Operational Pressure Index"].round(2)
)

df["Pressure Category"] = pd.cut(
    df["Operational Pressure Index"],
    bins=[-0.01, 25, 50, 75, 100],
    labels=["Low", "Moderate", "High", "Very High"],
    include_lowest=True
)

pressure_columns = [
    "YearMonth",
    "Passenger Count",
    "Landing Count",
    "Cargo Weight LBS",
    "Passenger Count Normalized",
    "Landing Count Normalized",
    "Cargo Weight LBS Normalized",
    "Operational Pressure Index",
    "Pressure Category"
]

pressure_data = df[pressure_columns].copy()

pressure_data.to_csv(
    output_folder / "integrated_operational_pressure.csv",
    index=False
)

print("\nPressure Index Summary:")
print(
    df["Operational Pressure Index"]
    .describe()
    .round(2)
    .to_string()
)

print("\nPressure Category Distribution:")
print(df["Pressure Category"].value_counts().to_string())

print("\n5. Year-over-Year Growth Analysis")

growth_columns = metrics + ["Total Landed Weight"]

growth = df[["YearMonth"] + growth_columns].copy()

for col in growth_columns:
    growth[col + " YoY Growth (%)"] = (
        growth[col].pct_change(periods=12, fill_method=None) * 100
    ).replace([float("inf"), float("-inf")], pd.NA)

growth.to_csv(
    output_folder / "integrated_yearly_growth.csv",
    index=False
)

print(growth.tail(12).round(2).to_string(index=False))

print("\n6. Peak and Lowest Operational Pressure")

valid_pressure = df.dropna(
    subset=["Operational Pressure Index"]
)

highest = valid_pressure.loc[
    valid_pressure["Operational Pressure Index"].idxmax()
]

lowest = valid_pressure.loc[
    valid_pressure["Operational Pressure Index"].idxmin()
]

print("\nHighest Pressure Month:")
print(highest["YearMonth"].strftime("%Y-%m"))
print("Index:", highest["Operational Pressure Index"])
print("Category:", highest["Pressure Category"])

print("\nLowest Pressure Month:")
print(lowest["YearMonth"].strftime("%Y-%m"))
print("Index:", lowest["Operational Pressure Index"])
print("Category:", lowest["Pressure Category"])

pressure_extremes = pd.DataFrame([
    {
        "Type": "Highest",
        "YearMonth": highest["YearMonth"].strftime("%Y-%m"),
        "Operational Pressure Index": highest["Operational Pressure Index"],
        "Pressure Category": highest["Pressure Category"]
    },
    {
        "Type": "Lowest",
        "YearMonth": lowest["YearMonth"].strftime("%Y-%m"),
        "Operational Pressure Index": lowest["Operational Pressure Index"],
        "Pressure Category": lowest["Pressure Category"]
    }
])

pressure_extremes.to_csv(
    output_folder / "integrated_pressure_extremes.csv",
    index=False
)

print("\n7. Monthly Seasonal Pressure")

seasonal = (
    df.groupby("Month")["Operational Pressure Index"]
    .mean()
    .reset_index()
)

seasonal.to_csv(
    output_folder / "integrated_seasonal_pressure.csv",
    index=False
)

print(seasonal.round(2).to_string(index=False))

print("\n8. Generated Reports")

for file in sorted(output_folder.glob("integrated_*.csv")):
    print(file.name)

print("\n========== INTEGRATED ANALYSIS COMPLETED ==========")
