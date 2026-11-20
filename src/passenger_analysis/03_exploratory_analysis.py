
import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

input_file = project_path / "data" / "processed" / "passenger_cleaned.csv"

output_folder = project_path / "reports" / "evaluation"
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

df["Activity Period Start Date"] = pd.to_datetime(
    df["Activity Period Start Date"]
)

df["Passenger Count"] = pd.to_numeric(
    df["Passenger Count"]
)

print("\n========== AEROCAST AI ==========")
print("PASSENGER EXPLORATORY DATA ANALYSIS")
print("=================================")

print("\n1. Overall Passenger Statistics")
print(df["Passenger Count"].describe())

print("\n2. Total Passenger Count")
print(df["Passenger Count"].sum())

print("\n3. Date Range")
print("Start Date:", df["Activity Period Start Date"].min())
print("End Date:", df["Activity Period Start Date"].max())

print("\n4. Passenger Activity Type")
activity_summary = df.groupby("Activity Type Code")[
    "Passenger Count"
].sum().sort_values(ascending=False)

print(activity_summary)

print("\n5. Top 10 Airlines")
airline_summary = df.groupby("Operating Airline")[
    "Passenger Count"
].sum().sort_values(ascending=False).head(10)

print(airline_summary)

print("\n6. Domestic and International Passengers")
geo_summary = df.groupby("GEO Summary")[
    "Passenger Count"
].sum().sort_values(ascending=False)

print(geo_summary)

print("\n7. Passenger Count by Terminal")
terminal_summary = df.groupby("Terminal")[
    "Passenger Count"
].sum().sort_values(ascending=False)

print(terminal_summary)

print("\n8. Passenger Count by Boarding Area")
boarding_summary = df.groupby("Boarding Area")[
    "Passenger Count"
].sum().sort_values(ascending=False)

print(boarding_summary)

print("\n9. Monthly Passenger Traffic")

monthly_summary = df.groupby(
    "Activity Period Start Date"
)["Passenger Count"].sum().sort_index()

print(monthly_summary.head(12))

print("\n10. Yearly Passenger Traffic")

df["Year"] = df["Activity Period Start Date"].dt.year

yearly_summary = df.groupby("Year")[
    "Passenger Count"
].sum()

print(yearly_summary)

print("\n11. Saving Analysis Results")

activity_summary.to_csv(output_folder / "passenger_activity_summary.csv")
airline_summary.to_csv(output_folder / "top_10_airlines.csv")
geo_summary.to_csv(output_folder / "domestic_international_summary.csv")
terminal_summary.to_csv(output_folder / "terminal_summary.csv")
boarding_summary.to_csv(output_folder / "boarding_area_summary.csv")
monthly_summary.to_csv(output_folder / "monthly_passenger_summary.csv")
yearly_summary.to_csv(output_folder / "yearly_passenger_summary.csv")

print("All analysis summaries saved successfully.")

print("\n========== EDA COMPLETED ==========")
