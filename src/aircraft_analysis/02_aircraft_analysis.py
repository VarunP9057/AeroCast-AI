import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

input_file = project_path / "datasets" / "Air_Traffic_Landings_Statistics_20261106.csv"

output_folder = project_path / "reports" / "evaluation"
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, low_memory=False)

df["Activity Period Start Date"] = pd.to_datetime(
    df["Activity Period Start Date"],
    errors="coerce"
)

df["Landing Count"] = pd.to_numeric(
    df["Landing Count"],
    errors="coerce"
)

df["Total Landed Weight"] = pd.to_numeric(
    df["Total Landed Weight"],
    errors="coerce"
)

print("\n========== AEROCAST AI ==========")
print("AIRCRAFT LANDINGS ANALYSIS")
print("================================")

print("\n1. Overall Aircraft Statistics")

print("Total Records:", len(df))
print("Total Landings:", df["Landing Count"].sum())
print("Total Landed Weight:", df["Total Landed Weight"].sum())

print("\n2. Landing Count Statistics")
print(df["Landing Count"].describe())

print("\n3. Top 10 Airlines by Landing Count")

airline_summary = df.groupby(
    "Operating Airline"
)["Landing Count"].sum().sort_values(ascending=False).head(10)

print(airline_summary)

print("\n4. Aircraft Type Analysis")

aircraft_summary = df.groupby(
    "Landing Aircraft Type"
)["Landing Count"].sum().sort_values(ascending=False)

print(aircraft_summary)

print("\n5. Aircraft Body Type Analysis")

body_summary = df.groupby(
    "Aircraft Body Type"
)["Landing Count"].sum().sort_values(ascending=False)

print(body_summary)

print("\n6. Aircraft Manufacturer Analysis")

manufacturer_summary = df.groupby(
    "Aircraft Manufacturer"
)["Landing Count"].sum().sort_values(ascending=False).head(10)

print(manufacturer_summary)

print("\n7. Top 10 Aircraft Models")

model_summary = df.groupby(
    "Aircraft Model"
)["Landing Count"].sum().sort_values(ascending=False).head(10)

print(model_summary)

print("\n8. Landing Weight Analysis")

weight_summary = df.groupby(
    "Aircraft Body Type"
)["Total Landed Weight"].sum().sort_values(ascending=False)

print(weight_summary)

print("\n9. Domestic and International Landings")

geo_summary = df.groupby(
    "GEO Summary"
)["Landing Count"].sum().sort_values(ascending=False)

print(geo_summary)

print("\n10. Monthly Aircraft Landings")

monthly_summary = df.groupby(
    "Activity Period Start Date"
)["Landing Count"].sum().sort_index()

print(monthly_summary.head(12))

print("\n11. Yearly Aircraft Landings")

df["Year"] = df["Activity Period Start Date"].dt.year

yearly_summary = df.groupby("Year")[
    "Landing Count"
].sum()

print(yearly_summary)

print("\n12. Saving Analysis Results")

airline_summary.to_csv(output_folder / "top_10_aircraft_airlines.csv")
aircraft_summary.to_csv(output_folder / "aircraft_type_summary.csv")
body_summary.to_csv(output_folder / "aircraft_body_type_summary.csv")
manufacturer_summary.to_csv(output_folder / "aircraft_manufacturer_summary.csv")
model_summary.to_csv(output_folder / "top_10_aircraft_models.csv")
weight_summary.to_csv(output_folder / "aircraft_weight_summary.csv")
geo_summary.to_csv(output_folder / "aircraft_geo_summary.csv")
monthly_summary.to_csv(output_folder / "monthly_aircraft_landings.csv")
yearly_summary.to_csv(output_folder / "yearly_aircraft_landings.csv")

print("All aircraft analysis results saved successfully.")

print("\n========== AIRCRAFT ANALYSIS COMPLETED ==========")

