
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

input_path = (
    project_path / "reports" / "evaluation"
    / "integrated_operational_pressure.csv"
)

output_folder = project_path / "reports" / "figures" / "integrated"
output_folder.mkdir(parents=True, exist_ok=True)

if not input_path.exists():
    raise FileNotFoundError(
        "Run 02_integrated_analysis.py before generating visualizations."
    )

df = pd.read_csv(input_path)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])

BG = "#080F1F"
PANEL = "#142238"
CYAN = "#00D9FF"
BLUE = "#647BFF"
TEXT = "#EAF4FF"
MUTED = "#A9B9CC"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": TEXT,
    "grid.color": "#33445D",
    "grid.alpha": 0.35
})


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        output_folder / filename,
        dpi=300,
        facecolor=BG,
        bbox_inches="tight"
    )
    plt.close()
    print("Created:", filename)


print("\n========== AEROCAST AI ==========")
print("INTEGRATED DATA VISUALIZATIONS")
print("================================")


# 1. Monthly Passenger Traffic

plt.figure(figsize=(13, 6))

plt.plot(
    df["YearMonth"],
    df["Passenger Count"],
    color=CYAN,
    linewidth=1.7
)

plt.title("SFO Monthly Passenger Traffic", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Passenger Count")
plt.grid(True)

save_chart("integrated_passenger_trend.png")


# 2. Monthly Aircraft Landings

plt.figure(figsize=(13, 6))

plt.plot(
    df["YearMonth"],
    df["Landing Count"],
    color=BLUE,
    linewidth=1.7
)

plt.title("SFO Monthly Aircraft Landings", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Landing Count")
plt.grid(True)

save_chart("integrated_aircraft_trend.png")


# 3. Monthly Cargo Weight

plt.figure(figsize=(13, 6))

plt.plot(
    df["YearMonth"],
    df["Cargo Weight LBS"],
    color="#F4B860",
    linewidth=1.7
)

plt.title("SFO Monthly Cargo Weight", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Cargo Weight (LBS)")
plt.grid(True)

save_chart("integrated_cargo_trend.png")


# 4. Operational Pressure Index

plt.figure(figsize=(13, 6))

plt.plot(
    df["YearMonth"],
    df["Operational Pressure Index"],
    color=CYAN,
    linewidth=1.8,
    label="Operational Pressure Index"
)

plt.axhline(25, color=MUTED, linestyle="--", alpha=0.6)
plt.axhline(50, color=BLUE, linestyle="--", alpha=0.6)
plt.axhline(75, color="#F4B860", linestyle="--", alpha=0.6)

plt.title("SFO Relative Operational Pressure Index", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Pressure Index (0–100)")
plt.ylim(0, 105)
plt.grid(True)
plt.legend(facecolor=PANEL, labelcolor=TEXT)

save_chart("integrated_operational_pressure.png")


# 5. Seasonal Operational Pressure

# 5. Seasonal Operational Pressure

df["Month"] = df["YearMonth"].dt.month

seasonal = (
    df.groupby("Month")["Operational Pressure Index"]
    .mean()
    .reindex(range(1, 13))
)

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(12, 6))

bars = plt.bar(
    month_names,
    seasonal.values,
    color=CYAN,
    edgecolor=BG
)

plt.title("Average Seasonal Operational Pressure", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Average Pressure Index")
plt.ylim(0, 100)
plt.grid(axis="y")

for bar, value in zip(bars, seasonal.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.1f}",
        ha="center",
        color=TEXT,
        fontsize=9
    )

save_chart("integrated_seasonal_pressure.png")

# 6. Top 10 Highest Pressure Months

top_pressure = (
    df.nlargest(10, "Operational Pressure Index")
    .sort_values("Operational Pressure Index")
)

labels = top_pressure["YearMonth"].dt.strftime("%Y-%m")

plt.figure(figsize=(12, 7))

bars = plt.barh(
    labels,
    top_pressure["Operational Pressure Index"],
    color=BLUE,
    edgecolor=BG
)

plt.title("Top 10 Highest Operational Pressure Months", fontsize=16)
plt.xlabel("Operational Pressure Index")
plt.ylabel("Year and Month")
plt.xlim(0, 105)
plt.grid(axis="x")

for bar, value in zip(
    bars, top_pressure["Operational Pressure Index"]
):
    plt.text(
        value + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        color=TEXT
    )

save_chart("integrated_top_pressure_months.png")


print("\nAll integrated visualizations generated.")
print("Output folder:", output_folder)
print("\n========== VISUALIZATION COMPLETED ==========")
