
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_ROOT / "datasets"
OUTPUT_FOLDER = PROJECT_ROOT / "reports" / "figures"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

files = list(DATA_FOLDER.glob("Air_Traffic_Landings_Statistics*.csv"))

if not files:
    raise FileNotFoundError("Aircraft landing dataset not found.")

df = pd.read_csv(files[0], low_memory=False)
df.columns = df.columns.str.strip()

NAVY = "#080F1F"
CYAN = "#00D9FF"
WHITE = "#EAF4FF"

plt.rcParams.update({
    "figure.facecolor": NAVY,
    "axes.facecolor": NAVY,
    "axes.edgecolor": CYAN,
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "text.color": WHITE,
    "grid.color": CYAN,
    "grid.alpha": 0.18,
    "font.size": 10
})


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FOLDER / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=NAVY
    )
    plt.close()
    print("Created:", filename)


def create_category_chart(column, title, filename, value_column):
    if column not in df.columns:
        print("Skipped:", column)
        return

    temp = df[[column, value_column]].copy()

    temp[value_column] = pd.to_numeric(
        temp[value_column].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    summary = (
        temp.dropna(subset=[column, value_column])
        .groupby(column)[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    if summary.empty:
        print("No data available:", column)
        return

    plt.figure(figsize=(11, 6))

    bars = plt.barh(
        summary.index.astype(str),
        summary.values,
        color=CYAN,
        edgecolor=NAVY
    )

    plt.bar_label(
        bars,
        padding=4,
        color=WHITE,
        fontsize=8,
        fmt="%.0f"
    )

    plt.title(title, fontsize=16, weight="bold")
    plt.xlabel(value_column)
    plt.ylabel(column)
    plt.grid(axis="x", linestyle="--")

    save_chart(filename)


print("\n========== AEROCAST AI ==========")
print("AIRCRAFT DATA VISUALIZATION")
print("================================")
print("Dataset shape:", df.shape)

DATE_COL = "Activity Period Start Date"
LANDING_COL = "Landing Count"
WEIGHT_COL = "Total Landed Weight"

required_columns = [DATE_COL, LANDING_COL, WEIGHT_COL]

for column in required_columns:
    if column not in df.columns:
        raise KeyError(f"Required column missing: {column}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

for column in [LANDING_COL, WEIGHT_COL]:
    df[column] = pd.to_numeric(
        df[column].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

df = df.dropna(subset=[DATE_COL, LANDING_COL]).copy()

df = df[df[LANDING_COL] >= 0]

df["YearMonth"] = df[DATE_COL].dt.to_period("M")
df["Year"] = df[DATE_COL].dt.year
df["Month"] = df[DATE_COL].dt.month

print("Valid aircraft records:", len(df))
print("Total landings:", int(df[LANDING_COL].sum()))

# 1. Monthly Aircraft Landing Trend

monthly = (
    df.groupby("YearMonth")[LANDING_COL]
    .sum()
    .sort_index()
)

plt.figure(figsize=(13, 6))

plt.plot(
    monthly.index.astype(str),
    monthly.values,
    color=CYAN,
    linewidth=2.5,
    marker="o",
    markersize=3
)

plt.fill_between(
    range(len(monthly)),
    monthly.values,
    color=CYAN,
    alpha=0.12
)

plt.title("Monthly Aircraft Landings at SFO", fontsize=17, weight="bold")
plt.xlabel("Month")
plt.ylabel("Landing Count")

plt.xticks(
    range(len(monthly)),
    monthly.index.astype(str),
    rotation=60,
    ha="right"
)

plt.grid(True, linestyle="--")
save_chart("aircraft_monthly_trend.png")


# 2. Yearly Aircraft Landing Trend

yearly = (
    df.groupby("Year")[LANDING_COL]
    .sum()
    .sort_index()
)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    yearly.index.astype(str),
    yearly.values,
    color=CYAN,
    edgecolor=NAVY
)

plt.bar_label(bars, padding=3, color=WHITE, fontsize=8)

plt.title("Yearly Aircraft Landings at SFO", fontsize=17, weight="bold")
plt.xlabel("Year")
plt.ylabel("Landing Count")
plt.grid(axis="y", linestyle="--")

save_chart("aircraft_yearly_trend.png")


# 3. Domestic vs International Aircraft Landings

create_category_chart(
    "GEO Summary",
    "Domestic vs International Aircraft Landings",
    "aircraft_geo_summary.png",
    LANDING_COL
)


# 4. Top 10 Airlines by Aircraft Landings

create_category_chart(
    "Operating Airline",
    "Top 10 Airlines by Aircraft Landings",
    "aircraft_top_airlines.png",
    LANDING_COL
)


# 5. Aircraft Landing Type Distribution

create_category_chart(
    "Landing Aircraft Type",
    "Aircraft Landing Type Distribution",
    "aircraft_landing_type.png",
    LANDING_COL
)


# 6. Aircraft Body Type Distribution

create_category_chart(
    "Aircraft Body Type",
    "Aircraft Body Type Distribution",
    "aircraft_body_type.png",
    LANDING_COL
)


# 7. Aircraft Manufacturer Analysis

create_category_chart(
    "Aircraft Manufacturer",
    "Top 10 Aircraft Manufacturers by Landings",
    "aircraft_manufacturers.png",
    LANDING_COL
)


# 8. Aircraft Model Analysis

create_category_chart(
    "Aircraft Model",
    "Top 10 Aircraft Models by Landings",
    "aircraft_top_models.png",
    LANDING_COL
)


# 9. Total Landed Weight by Aircraft Body Type

weight_summary = (
    df.dropna(subset=[WEIGHT_COL, "Aircraft Body Type"])
    .groupby("Aircraft Body Type")[WEIGHT_COL]
    .sum()
    .sort_values()
)

if not weight_summary.empty:
    plt.figure(figsize=(11, 6))

    bars = plt.barh(
        weight_summary.index.astype(str),
        weight_summary.values,
        color=CYAN,
        edgecolor=NAVY
    )

    plt.bar_label(bars, padding=4, color=WHITE, fontsize=8)

    plt.title(
        "Total Landed Weight by Aircraft Body Type",
        fontsize=16,
        weight="bold"
    )
    plt.xlabel("Total Landed Weight")
    plt.ylabel("Aircraft Body Type")
    plt.grid(axis="x", linestyle="--")

    save_chart("aircraft_landed_weight.png")


# 10. Seasonal Aircraft Landing Trends

seasonal = (
    df.groupby("Month")[LANDING_COL]
    .mean()
    .reindex(range(1, 13))
)

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(11, 6))

plt.plot(
    month_names,
    seasonal.values,
    color=CYAN,
    linewidth=2.5,
    marker="o",
    markersize=7
)

plt.fill_between(
    range(12),
    seasonal.values,
    color=CYAN,
    alpha=0.12
)

plt.title("Average Monthly Seasonal Aircraft Landings", fontsize=16, weight="bold")
plt.xlabel("Month")
plt.ylabel("Average Landing Count")
plt.grid(True, linestyle="--")

save_chart("aircraft_seasonal_trend.png")


# 11. Peak and Lowest Aircraft Landing Months

peak_months = monthly.sort_values(ascending=False).head(10)
lowest_months = monthly.sort_values().head(10)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(
    peak_months.index.astype(str)[::-1],
    peak_months.values[::-1],
    color=CYAN
)

axes[0].set_title("Top 10 Peak Landing Months")
axes[0].set_xlabel("Landing Count")
axes[0].grid(axis="x", linestyle="--")

axes[1].barh(
    lowest_months.index.astype(str)[::-1],
    lowest_months.values[::-1],
    color=CYAN
)

axes[1].set_title("10 Lowest Landing Months")
axes[1].set_xlabel("Landing Count")
axes[1].grid(axis="x", linestyle="--")

save_chart("aircraft_peak_lowest_months.png")


print("\n========== AIRCRAFT ANALYTICS SUMMARY ==========")
print("Total landings:", int(df[LANDING_COL].sum()))
print("Total landed weight:", df[WEIGHT_COL].sum())
print("Highest landing month:", monthly.idxmax(), int(monthly.max()))
print("Lowest landing month:", monthly.idxmin(), int(monthly.min()))
print("Highest landing year:", yearly.idxmax(), int(yearly.max()))
print("Lowest landing year:", yearly.idxmin(), int(yearly.min()))

print("\nCharts saved in:", OUTPUT_FOLDER)
print("\n========== AIRCRAFT VISUALIZATION COMPLETED ==========")
