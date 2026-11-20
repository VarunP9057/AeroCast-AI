
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

dataset_path = project_path / "data" / "processed" / "passenger_cleaned.csv"
output_folder = project_path / "reports" / "figures"
output_folder.mkdir(parents=True, exist_ok=True)

if not dataset_path.exists():
    raise FileNotFoundError(
        "Passenger cleaned dataset not found. Run 02_clean_data.py first."
    )

df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()

DATE_COL = "Activity Period Start Date"
PASSENGER_COL = "Passenger Count"

required_columns = [DATE_COL, PASSENGER_COL]

for column in required_columns:
    if column not in df.columns:
        raise KeyError(f"Required column not found: {column}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

df[PASSENGER_COL] = pd.to_numeric(
    df[PASSENGER_COL].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

df = df.dropna(subset=[DATE_COL, PASSENGER_COL])
df = df[df[PASSENGER_COL] >= 0].copy()

df["YearMonth"] = df[DATE_COL].dt.to_period("M")
df["Year"] = df[DATE_COL].dt.year
df["Month"] = df[DATE_COL].dt.month

NAVY = "#080F1F"
PANEL = "#142238"
CYAN = "#00D9FF"
BLUE = "#647BFF"
WHITE = "#EAF4FF"

plt.rcParams.update({
    "figure.facecolor": NAVY,
    "axes.facecolor": PANEL,
    "axes.edgecolor": CYAN,
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "text.color": WHITE,
    "grid.color": CYAN,
    "grid.alpha": 0.18,
    "font.size": 10,
    "axes.titleweight": "bold"
})


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        output_folder / filename,
        dpi=300,
        facecolor=NAVY,
        bbox_inches="tight"
    )
    plt.close()
    print("Created:", filename)


def create_category_chart(column, title, filename):
    if column not in df.columns:
        print("Skipped:", column, "- column not found")
        return

    summary = (
        df.assign(**{column: df[column].fillna("Unknown").astype(str)})
        .groupby(column)[PASSENGER_COL]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    if summary.empty:
        print("Skipped:", filename, "- no data")
        return

    plt.figure(figsize=(11, 6))

    bars = plt.barh(
        summary.index,
        summary.values,
        color=CYAN,
        edgecolor=NAVY
    )

    plt.bar_label(
        bars,
        labels=[f"{value:,.0f}" for value in summary.values],
        padding=4,
        color=WHITE,
        fontsize=8
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Passenger Count")
    plt.ylabel(column)
    plt.grid(axis="x", linestyle="--")

    save_chart(filename)


print("\n========== AEROCAST AI ==========")
print("PASSENGER DATA VISUALIZATION")
print("================================")
print("Dataset shape:", df.shape)

# 1. Total passenger traffic and monthly trend

monthly = (
    df.groupby("YearMonth")[PASSENGER_COL]
    .sum()
    .sort_index()
)

if not monthly.empty:
    x = range(len(monthly))

    plt.figure(figsize=(14, 6))

    plt.plot(
        x,
        monthly.values,
        color=CYAN,
        linewidth=2,
        marker="o",
        markersize=2
    )

    plt.fill_between(
        x,
        monthly.values,
        color=CYAN,
        alpha=0.12
    )

    tick_positions = list(range(0, len(monthly), 12))
    if len(monthly) - 1 not in tick_positions:
        tick_positions.append(len(monthly) - 1)

    tick_labels = [
        str(monthly.index[position])
        for position in tick_positions
    ]

    plt.xticks(
        tick_positions,
        tick_labels,
        rotation=45,
        ha="right"
    )

    plt.title("Monthly Passenger Traffic at SFO", fontsize=17)
    plt.xlabel("Year-Month")
    plt.ylabel("Passenger Count")
    plt.grid(True, linestyle="--")

    save_chart("passenger_monthly_trend.png")


# 2. Year-wise passenger traffic

yearly = (
    df.groupby("Year")[PASSENGER_COL]
    .sum()
    .sort_index()
)

if not yearly.empty:
    plt.figure(figsize=(13, 6))

    bars = plt.bar(
        yearly.index.astype(str),
        yearly.values,
        color=CYAN,
        edgecolor=NAVY
    )

    plt.bar_label(
        bars,
        labels=[f"{value:,.0f}" for value in yearly.values],
        padding=3,
        color=WHITE,
        fontsize=7,
        rotation=90
    )

    plt.title("Yearly Passenger Traffic at SFO", fontsize=17)
    plt.xlabel("Year")
    plt.ylabel("Passenger Count")
    plt.xticks(rotation=60)
    plt.grid(axis="y", linestyle="--")

    save_chart("passenger_yearly_trend.png")


# 3. Year-over-year passenger growth

yearly_growth = yearly.pct_change() * 100

growth_data = yearly_growth.dropna()

if not growth_data.empty:
    plt.figure(figsize=(13, 6))

    colors = [
        CYAN if value >= 0 else BLUE
        for value in growth_data.values
    ]

    bars = plt.bar(
        growth_data.index.astype(str),
        growth_data.values,
        color=colors,
        edgecolor=NAVY
    )

    plt.axhline(0, color=WHITE, linewidth=0.8)

    plt.title("Year-over-Year Passenger Growth", fontsize=17)
    plt.xlabel("Year")
    plt.ylabel("Growth (%)")
    plt.xticks(rotation=60)
    plt.grid(axis="y", linestyle="--")

    save_chart("passenger_yearly_growth.png")


# 4. Domestic vs International passenger traffic

create_category_chart(
    "GEO Summary",
    "Domestic vs International Passenger Traffic",
    "passenger_geo_summary.png"
)


# 5. Airline ranking

create_category_chart(
    "Operating Airline",
    "Top 10 Airlines by Passenger Traffic",
    "passenger_top_airlines.png"
)


# 6. Terminal distribution

create_category_chart(
    "Terminal",
    "Passenger Traffic by Terminal",
    "passenger_terminal.png"
)


# 7. Boarding area distribution

create_category_chart(
    "Boarding Area",
    "Passenger Traffic by Boarding Area",
    "passenger_boarding_area.png"
)


# 8. Enplaned, Deplaned and Thru / Transit analysis

activity_col = "Activity Type Code"

if activity_col in df.columns:
    activity_summary = (
        df.assign(
            **{
                activity_col: df[activity_col].fillna("Unknown").astype(str)
            }
        )
        .groupby(activity_col)[PASSENGER_COL]
        .sum()
        .sort_values(ascending=False)
    )

    if not activity_summary.empty:
        plt.figure(figsize=(10, 6))

        bars = plt.bar(
            activity_summary.index,
            activity_summary.values,
            color=[CYAN, BLUE, "#3B82A0"][:len(activity_summary)],
            edgecolor=NAVY
        )

        plt.bar_label(
            bars,
            labels=[
                f"{value:,.0f}"
                for value in activity_summary.values
            ],
            padding=4,
            color=WHITE
        )

        plt.title(
            "Enplaned vs Deplaned vs Thru / Transit Passengers",
            fontsize=15
        )
        plt.xlabel("Activity Type")
        plt.ylabel("Passenger Count")
        plt.grid(axis="y", linestyle="--")

        save_chart("passenger_activity_type.png")


# 9. Seasonal passenger traffic

seasonal = (
    df.groupby("Month")[PASSENGER_COL]
    .mean()
    .reindex(range(1, 13))
)

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

if not seasonal.dropna().empty:
    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        month_names,
        seasonal.values,
        color=CYAN,
        edgecolor=NAVY
    )

    plt.bar_label(
        bars,
        labels=[
            f"{value:,.0f}" if pd.notna(value) else ""
            for value in seasonal.values
        ],
        padding=3,
        color=WHITE,
        fontsize=8
    )

    plt.title(
        "Average Monthly Passenger Traffic - Seasonal Pattern",
        fontsize=16
    )
    plt.xlabel("Calendar Month")
    plt.ylabel("Average Passenger Count")
    plt.grid(axis="y", linestyle="--")

    save_chart("passenger_seasonal_trend.png")


# 10. Peak and lowest passenger traffic months

if not monthly.empty:
    peak_months = monthly.nlargest(10)
    lowest_months = monthly.nsmallest(10)

    peak_lowest = pd.concat([
        lowest_months,
        peak_months
    ]).sort_values()

    plt.figure(figsize=(12, 8))

    colors = [
        BLUE if value in lowest_months.values else CYAN
        for value in peak_lowest.values
    ]

    bars = plt.barh(
        peak_lowest.index.astype(str),
        peak_lowest.values,
        color=colors,
        edgecolor=NAVY
    )

    plt.bar_label(
        bars,
        labels=[
            f"{value:,.0f}"
            for value in peak_lowest.values
        ],
        padding=4,
        color=WHITE,
        fontsize=8
    )

    plt.title(
        "Top 10 Peak and Lowest Passenger Traffic Months",
        fontsize=16
    )
    plt.xlabel("Passenger Count")
    plt.ylabel("Year-Month")
    plt.grid(axis="x", linestyle="--")

    save_chart("passenger_peak_lowest_months.png")


# Passenger analytics summary

total_passengers = df[PASSENGER_COL].sum()
average_monthly = monthly.mean() if not monthly.empty else 0

print("\n========== PASSENGER ANALYTICS SUMMARY ==========")
print(f"Total passenger records: {len(df):,}")
print(f"Total passenger traffic: {total_passengers:,.0f}")
print(f"Average monthly passenger traffic: {average_monthly:,.0f}")

if not monthly.empty:
    print(
        "Highest passenger month:",
        monthly.idxmax(),
        f"{monthly.max():,.0f}"
    )

    print(
        "Lowest passenger month:",
        monthly.idxmin(),
        f"{monthly.min():,.0f}"
    )

if not yearly.empty:
    print(
        "Highest passenger year:",
        yearly.idxmax(),
        f"{yearly.max():,.0f}"
    )

if "Activity Type Code" in df.columns:
    print("\nActivity Type Passenger Totals:")
    print(
        df.groupby("Activity Type Code")[PASSENGER_COL]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

print("\nCharts saved in:", output_folder)
print("\n========== PASSENGER VISUALIZATION COMPLETED ==========")

