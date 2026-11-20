
import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]
dataset_folder = project_path / "datasets"
output_folder = project_path / "reports" / "evaluation"

output_folder.mkdir(parents=True, exist_ok=True)

files = list(dataset_folder.glob("Air_Traffic_Cargo_Statistics*.csv"))

if not files:
    raise FileNotFoundError("Cargo dataset not found.")

df = pd.read_csv(files[0], low_memory=False)

df.columns = df.columns.str.strip()

print("\n========== AEROCAST AI ==========")
print("CARGO DATA ANALYSIS")
print("================================")

print("\n1. Dataset Overview")
print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

date_col = next(
    (c for c in df.columns if "date" in c.lower() or "period" in c.lower()),
    None
)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month

numeric_columns = df.select_dtypes(include="number").columns.tolist()

print("\n2. Numeric Columns")
print(numeric_columns)

for col in numeric_columns:
    print("\n", col)
    print("Total:", df[col].sum())
    print("Average:", round(df[col].mean(), 2))
    print("Maximum:", df[col].max())
    print("Minimum:", df[col].min())

df.to_csv(output_folder / "cargo_numeric_summary_source.csv", index=False)

category_columns = [
    c for c in df.select_dtypes(include="object").columns
    if c not in ["Year", "Month"]
]

for col in category_columns:
    summary = df[col].value_counts().reset_index()
    summary.columns = [col, "Record Count"]
    summary.to_csv(
        output_folder / f"cargo_{col.lower().replace(' ', '_')}_summary.csv",
        index=False
    )

    print(f"\n3. Top 10 Categories - {col}")
    print(summary.head(10).to_string(index=False))

if date_col:
    monthly = (
        df.dropna(subset=[date_col])
        .groupby([ "Year", "Month" ])
        .size()
        .reset_index(name="Record Count")
        .sort_values(["Year", "Month"])
    )

    yearly = (
        df.dropna(subset=[date_col])
        .groupby("Year")
        .size()
        .reset_index(name="Record Count")
    )

    monthly.to_csv(output_folder / "cargo_monthly_records.csv", index=False)
    yearly.to_csv(output_folder / "cargo_yearly_records.csv", index=False)

    print("\n4. Monthly Cargo Records")
    print(monthly.tail(12).to_string(index=False))

    print("\n5. Yearly Cargo Records")
    print(yearly.to_string(index=False))

print("\n6. Generated Reports")
for file in sorted(output_folder.glob("cargo_*.csv")):
    print(file.name)

print("\n========== CARGO ANALYSIS COMPLETED ==========")
