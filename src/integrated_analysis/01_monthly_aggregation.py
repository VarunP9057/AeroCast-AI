
import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

dataset_folder = project_path / "datasets"
output_folder = project_path / "data" / "processed"

output_folder.mkdir(parents=True, exist_ok=True)


def load_dataset(pattern):
    files = list(dataset_folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Dataset not found: {pattern}")

    df = pd.read_csv(files[0], low_memory=False)
    df.columns = df.columns.str.strip()

    return df


def prepare_monthly_data(df, value_columns, output_name):
    date_col = next(
        (
            col for col in df.columns
            if "date" in col.lower()
        ),
        None
    )

    if date_col is None:
        raise ValueError("No date column found.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[date_col]).copy()

    df["YearMonth"] = df[date_col].dt.to_period("M")

    available_columns = [
        col for col in value_columns
        if col in df.columns
    ]

    for col in available_columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    if available_columns:
        monthly = (
            df.groupby("YearMonth")[available_columns]
            .sum(min_count=1)
            .reset_index()
        )
    else:
        monthly = (
            df.groupby("YearMonth")
            .size()
            .reset_index(name="Cargo Record Count")
        )

    monthly["Year"] = monthly["YearMonth"].dt.year
    monthly["Month"] = monthly["YearMonth"].dt.month
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    monthly.to_csv(output_folder / output_name, index=False)

    print(f"\nCreated: {output_name}")
    print("Monthly records:", len(monthly))
    print(monthly.head())

    return monthly


print("\n========== AEROCAST AI ==========")
print("INTEGRATED MONTHLY AGGREGATION")
print("================================")


# 1. Passenger Data

passenger_df = load_dataset("Air_Traffic_Passenger_Statistics*.csv")

passenger_monthly = prepare_monthly_data(
    passenger_df,
    ["Passenger Count"],
    "passenger_monthly.csv"
)


# 2. Aircraft Data

aircraft_df = load_dataset("Air_Traffic_Landings_Statistics*.csv")

aircraft_monthly = prepare_monthly_data(
    aircraft_df,
    ["Landing Count", "Total Landed Weight"],
    "aircraft_monthly.csv"
)


# 3. Cargo Data

cargo_df = load_dataset("Air_Traffic_Cargo_Statistics*.csv")

cargo_metrics = [
    col for col in cargo_df.columns
    if (
        "weight" in col.lower()
        or "cargo count" in col.lower()
        or "cargo volume" in col.lower()
    )
    and pd.api.types.is_numeric_dtype(
        pd.to_numeric(
            cargo_df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
    )
]

print("\nCargo metric columns detected:", cargo_metrics)

cargo_monthly = prepare_monthly_data(
    cargo_df,
    cargo_metrics,
    "cargo_monthly.csv"
)


# 4. Combine Monthly Datasets

integrated = passenger_monthly.merge(
    aircraft_monthly,
    on=["YearMonth", "Year", "Month"],
    how="outer"
)

integrated = integrated.merge(
    cargo_monthly,
    on=["YearMonth", "Year", "Month"],
    how="outer"
)

integrated = integrated.sort_values(
    ["Year", "Month"]
).reset_index(drop=True)

integrated.to_csv(
    output_folder / "integrated_monthly_airport_data.csv",
    index=False
)


print("\n========== INTEGRATED DATASET ==========")
print("Total Months:", len(integrated))
print("\nColumn Names:")
print(integrated.columns.tolist())

print("\nFirst 10 Records:")
print(integrated.head(10).to_string(index=False))

print("\nMissing Values:")
print(integrated.isnull().sum())

print("\nSaved: integrated_monthly_airport_data.csv")
print("\n========== TASK 11 COMPLETED ==========")
