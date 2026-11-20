
import pandas as pd
from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_PATH / "data" / "processed"


def load_csv(filename):
    file_path = DATA_FOLDER / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {filename}")

    return pd.read_csv(file_path)


# 1. Passenger Analytics
def get_passenger_analytics():

    df = load_csv("passenger_monthly.csv")

    total_passengers = int(df["Passenger Count"].sum())
    average_passengers = round(df["Passenger Count"].mean(), 2)

    highest_row = df.loc[df["Passenger Count"].idxmax()]
    lowest_row = df.loc[df["Passenger Count"].idxmin()]
    latest_row = df.iloc[-1]

    return {
        "total_passengers": total_passengers,
        "average_monthly_passengers": average_passengers,
        "highest_month": str(highest_row["YearMonth"]),
        "highest_passengers": int(highest_row["Passenger Count"]),
        "lowest_month": str(lowest_row["YearMonth"]),
        "lowest_passengers": int(lowest_row["Passenger Count"]),
        "latest_month": str(latest_row["YearMonth"]),
        "latest_passengers": int(latest_row["Passenger Count"]),
        "total_records": len(df)
    }


# 2. Aircraft Landing Analytics
def get_aircraft_analytics():

    df = load_csv("aircraft_monthly.csv")

    total_landings = int(df["Landing Count"].sum())
    total_weight = float(df["Total Landed Weight"].sum())
    average_landings = round(df["Landing Count"].mean(), 2)

    highest_row = df.loc[df["Landing Count"].idxmax()]
    latest_row = df.iloc[-1]

    return {
        "total_landings": total_landings,
        "total_landed_weight": round(total_weight, 2),
        "average_monthly_landings": average_landings,
        "highest_landing_month": str(highest_row["YearMonth"]),
        "highest_landings": int(highest_row["Landing Count"]),
        "latest_month": str(latest_row["YearMonth"]),
        "latest_landings": int(latest_row["Landing Count"]),
        "total_records": len(df)
    }


# 2A. Monthly Aircraft Analytics
def get_aircraft_monthly_analytics():

    df = load_csv("aircraft_monthly.csv")

    total_landings = int(df["Landing Count"].sum())
    total_weight = float(df["Total Landed Weight"].sum())
    average_landings = round(df["Landing Count"].mean(), 2)

    monthly_traffic = [
        {
            "month": str(row["YearMonth"]),
            "landings": int(row["Landing Count"]),
            "landed_weight": float(row["Total Landed Weight"])
        }
        for _, row in df.iterrows()
    ]

    return {
        "total_landings": total_landings,
        "total_landed_weight": round(total_weight, 2),
        "average_monthly_landings": average_landings,
        "monthly_traffic": monthly_traffic
    }


# 3. Cargo Analytics
def get_cargo_analytics():

    df = load_csv("cargo_monthly.csv")

    total_cargo = float(df["Cargo Weight LBS"].sum())
    average_cargo = round(df["Cargo Weight LBS"].mean(), 2)

    highest_row = df.loc[df["Cargo Weight LBS"].idxmax()]
    latest_row = df.iloc[-1]

    return {
        "total_cargo_weight_lbs": round(total_cargo, 2),
        "average_monthly_cargo_lbs": average_cargo,
        "highest_cargo_month": str(highest_row["YearMonth"]),
        "highest_cargo_weight_lbs": float(highest_row["Cargo Weight LBS"]),
        "latest_month": str(latest_row["YearMonth"]),
        "latest_cargo_weight_lbs": float(latest_row["Cargo Weight LBS"]),
        "total_records": len(df)
    }


# 4. Integrated Airport Analytics
def get_integrated_analytics():

    df = load_csv("integrated_monthly_airport_data.csv")

    total_passengers = int(df["Passenger Count"].sum())
    total_landings = int(df["Landing Count"].sum())
    total_cargo = float(df["Cargo Weight LBS"].sum())
    total_landed_weight = float(df["Total Landed Weight"].sum())

    latest_row = df.iloc[-1]

    return {
        "total_passengers": total_passengers,
        "total_landings": total_landings,
        "total_cargo_weight_lbs": round(total_cargo, 2),
        "total_landed_weight": round(total_landed_weight, 2),
        "latest_month": str(latest_row["YearMonth"]),
        "latest_passengers": int(latest_row["Passenger Count"]),
        "latest_landings": int(latest_row["Landing Count"]),
        "latest_cargo_weight_lbs": float(latest_row["Cargo Weight LBS"]),
        "total_records": len(df)
    }


# 5. Detailed Passenger Analytics
def get_detailed_passenger_analytics():

    df = load_csv("passenger_cleaned.csv")

    df["Activity Period Start Date"] = pd.to_datetime(
        df["Activity Period Start Date"],
        errors="coerce"
    )

    df = df.dropna(subset=["Activity Period Start Date"])

    df["Year"] = df["Activity Period Start Date"].dt.year
    df["Month"] = df["Activity Period Start Date"].dt.month
    df["YearMonth"] = df["Activity Period Start Date"].dt.strftime("%Y-%m")

    # Monthly Passenger Traffic
    monthly = (
        df.groupby("YearMonth")["Passenger Count"]
        .sum()
        .reset_index()
        .sort_values("YearMonth")
    )

    # Year-wise Passenger Traffic and Growth
    yearly = (
        df.groupby("Year")["Passenger Count"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    yearly["Growth Percentage"] = (
        yearly["Passenger Count"].pct_change() * 100
    )

    yearly_data = [
        {
            "year": int(row["Year"]),
            "passengers": int(row["Passenger Count"]),
            "growth_percentage": (
                None if pd.isna(row["Growth Percentage"])
                else round(float(row["Growth Percentage"]), 2)
            )
        }
        for _, row in yearly.iterrows()
    ]

    # Domestic vs International
    geography = (
        df.groupby("GEO Summary")["Passenger Count"]
        .sum()
        .sort_values(ascending=False)
    )

    # Airline Rankings - Top 10
    airlines = (
        df.groupby("Operating Airline")["Passenger Count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # Terminal Distribution
    terminals = (
        df.groupby("Terminal")["Passenger Count"]
        .sum()
        .sort_values(ascending=False)
    )

    # Boarding Area Distribution
    boarding_areas = (
        df.groupby("Boarding Area")["Passenger Count"]
        .sum()
        .sort_values(ascending=False)
    )

    # Enplaned, Deplaned and Transit
    activity = (
        df.groupby("Activity Type Code")["Passenger Count"]
        .sum()
        .sort_values(ascending=False)
    )

    # Seasonal Passenger Traffic
    yearly_monthly = (
        df.groupby(["Year", "Month"])["Passenger Count"]
        .sum()
        .reset_index()
    )

    seasonal = (
        yearly_monthly.groupby("Month")["Passenger Count"]
        .mean()
        .reset_index()
        .sort_values("Month")
    )

    seasonal_data = [
        {
            "month": int(row["Month"]),
            "month_name": pd.Timestamp(
                year=2000,
                month=int(row["Month"]),
                day=1
            ).strftime("%B"),
            "average_passengers": round(
                float(row["Passenger Count"]), 2
            )
        }
        for _, row in seasonal.iterrows()
    ]

    # Peak and Lowest Months
    highest_month = monthly.loc[
        monthly["Passenger Count"].idxmax()
    ]

    lowest_month = monthly.loc[
        monthly["Passenger Count"].idxmin()
    ]

    # Final Detailed Passenger Analytics Response
    return {
        "total_passengers": int(df["Passenger Count"].sum()),

        "monthly_traffic": [
            {
                "month": str(row["YearMonth"]),
                "passengers": int(row["Passenger Count"])
            }
            for _, row in monthly.iterrows()
        ],

        "yearly_traffic": yearly_data,

        "domestic_international": [
            {
                "category": str(name),
                "passengers": int(value)
            }
            for name, value in geography.items()
        ],

        "airline_rankings": [
            {
                "airline": str(name),
                "passengers": int(value)
            }
            for name, value in airlines.items()
        ],

        "terminal_distribution": [
            {
                "terminal": str(name),
                "passengers": int(value)
            }
            for name, value in terminals.items()
        ],

        "boarding_area_distribution": [
            {
                "boarding_area": str(name),
                "passengers": int(value)
            }
            for name, value in boarding_areas.items()
        ],

        "activity_distribution": [
            {
                "activity": str(name),
                "passengers": int(value)
            }
            for name, value in activity.items()
        ],

        "seasonal_traffic": seasonal_data,

        "peak_month": {
            "month": str(highest_month["YearMonth"]),
            "passengers": int(highest_month["Passenger Count"])
        },

        "lowest_month": {
            "month": str(lowest_month["YearMonth"]),
            "passengers": int(lowest_month["Passenger Count"])
        }
    }

# 6. Monthly Cargo Analytics
def get_cargo_monthly_analytics():

    df = load_csv("cargo_monthly.csv")

    monthly_traffic = [
        {
            "month": str(row["YearMonth"]),
            "cargo_weight_lbs": float(row["Cargo Weight LBS"])
        }
        for _, row in df.iterrows()
    ]

    return {
        "monthly_traffic": monthly_traffic
    }
