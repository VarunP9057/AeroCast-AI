
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "datasets" / "Air_Traffic_Cargo_Statistics_20261106.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "reports" / "figures"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, low_memory=False)

NAVY = "#080F1F"
CYAN = "#00D9FF"
WHITE = "#EAF4FF"

plt.rcParams["figure.facecolor"] = NAVY
plt.rcParams["axes.facecolor"] = NAVY
plt.rcParams["axes.labelcolor"] = WHITE
plt.rcParams["xtick.color"] = WHITE
plt.rcParams["ytick.color"] = WHITE
plt.rcParams["text.color"] = WHITE
plt.rcParams["axes.edgecolor"] = CYAN

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


print("\n========== AEROCAST AI ==========")
print("CARGO DATA VISUALIZATION")
print("================================")
print("Dataset shape:", df.shape)

date_col = "Activity Period Start Date"
weight_col = "Cargo Weight LBS"

if date_col in df.columns and weight_col in df.columns:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df[weight_col] = pd.to_numeric(
        df[weight_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    date_data = df.dropna(
        subset=[date_col, weight_col]
    ).copy()

    date_data = date_data[date_data[weight_col] >= 0]

    date_data["YearMonth"] = date_data[date_col].dt.to_period("M")

    monthly = (
        date_data.groupby("YearMonth")[weight_col]
        .sum()
        .sort_index()
    )

    if not monthly.empty:
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

        plt.title("Monthly Cargo Weight at SFO", fontsize=17, weight="bold")
        plt.xlabel("Month")
        plt.ylabel("Cargo Weight (LBS)")

        plt.xticks(
            range(len(monthly)),
            monthly.index.astype(str),
            rotation=60,
            ha="right"
        )

        plt.grid(True, linestyle="--", alpha=0.3)
        save_chart("cargo_monthly_trend.png")

    yearly = (
        date_data.groupby(date_data[date_col].dt.year)[weight_col]
        .sum()
        .sort_index()
    )

    if not yearly.empty:
        plt.figure(figsize=(11, 6))

        bars = plt.bar(
            yearly.index.astype(str),
            yearly.values,
            color=CYAN,
            edgecolor=NAVY
        )

        plt.bar_label(
            bars,
            padding=3,
            color=WHITE,
            fontsize=8,
            fmt="%.0f"
        )

        plt.title("Yearly Cargo Weight at SFO", fontsize=17, weight="bold")
        plt.xlabel("Year")
        plt.ylabel("Cargo Weight (LBS)")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        save_chart("cargo_yearly_trend.png")

else:
    print("Required date or cargo weight column not found.")

def create_category_chart(column, title, filename):

    if column not in df.columns:
        print("Skipped:", column, "- column not found")
        return

    temp = df[[column, weight_col]].copy()

    temp[weight_col] = pd.to_numeric(
        temp[weight_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

    summary = (
        temp.dropna(subset=[column, weight_col])
        .groupby(column)[weight_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    if summary.empty:
        print("Skipped:", filename, "- no valid data")
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
        padding=3,
        color=WHITE,
        fontsize=8,
        fmt="%.0f"
    )

    plt.title(title, fontsize=16, weight="bold")
    plt.xlabel("Cargo Weight (LBS)")
    plt.ylabel(column)
    plt.grid(axis="x", linestyle="--", alpha=0.3)

    save_chart(filename)


print("\n========== CATEGORY ANALYSIS ==========")

create_category_chart(
    "Operating Airline",
    "Top 10 Airlines by Cargo Weight",
    "cargo_top_airlines.png"
)

create_category_chart(
    "GEO Summary",
    "Domestic vs International Cargo Weight",
    "cargo_geo_summary.png"
)

create_category_chart(
    "Cargo Type Code",
    "Cargo Weight by Cargo Type",
    "cargo_type_analysis.png"
)

create_category_chart(
    "Cargo Aircraft Type",
    "Cargo Weight by Aircraft Type",
    "cargo_aircraft_type.png"
)

print("\nCharts saved in:", OUTPUT_FOLDER)
print("\n========== CARGO VISUALIZATION COMPLETED ==========")
