
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

input_path = (
    project_path / "data" / "processed"
    / "integrated_monthly_airport_data.csv"
)

output_folder = project_path / "reports" / "evaluation"
figure_folder = project_path / "reports" / "figures" / "integrated"

output_folder.mkdir(parents=True, exist_ok=True)
figure_folder.mkdir(parents=True, exist_ok=True)

if not input_path.exists():
    raise FileNotFoundError(
        "Integrated dataset not found."
    )

df = pd.read_csv(input_path)

print("\n========== AEROCAST AI ==========")
print("INTEGRATED CORRELATION ANALYSIS")
print("================================")

metrics = [
    "Passenger Count",
    "Landing Count",
    "Cargo Weight LBS",
    "Total Landed Weight"
]

missing_columns = [
    col for col in metrics if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

correlation_data = df[metrics].apply(
    pd.to_numeric, errors="coerce"
).dropna()

correlation = correlation_data.corr(method="pearson")

print("\n1. Selected Airport Metrics")
print(metrics)

print("\n2. Pearson Correlation Matrix")
print(correlation.round(3).to_string())

correlation.to_csv(
    output_folder / "airport_correlation_matrix.csv"
)

# Pairwise relationships

relationships = []

for i in range(len(metrics)):
    for j in range(i + 1, len(metrics)):
        value = correlation.iloc[i, j]

        relationships.append({
            "Metric 1": metrics[i],
            "Metric 2": metrics[j],
            "Correlation": round(value, 3)
        })

relationships_df = pd.DataFrame(relationships)

relationships_df["Absolute Correlation"] = (
    relationships_df["Correlation"].abs()
)

relationships_df = relationships_df.sort_values(
    "Absolute Correlation",
    ascending=False
)

relationships_df.to_csv(
    output_folder / "airport_metric_relationships.csv",
    index=False
)

print("\n3. Strongest Metric Relationships")

print(
    relationships_df.drop(
        columns=["Absolute Correlation"]
    ).to_string(index=False)
)

# Midnight Blue / Electric Cyan Theme

BG = "#080F1F"
PANEL = "#142238"
CYAN = "#00D9FF"
TEXT = "#EAF4FF"
MUTED = "#A9B9CC"

plt.figure(
    figsize=(12, 9),
    facecolor=BG
)

axes = sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=1.2,
    linecolor=BG,
    fmt=".2f",
    square=True,
    annot_kws={"size": 12, "weight": "bold"},
    cbar_kws={"label": "Pearson Correlation"}
)

axes.set_facecolor(PANEL)

axes.set_title(
    "SFO Airport Metrics Correlation Heatmap",
    color=TEXT,
    fontsize=16,
    fontweight="bold",
    pad=20
)

axes.tick_params(colors=TEXT, labelsize=10)

plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)

colorbar = axes.collections[0].colorbar
colorbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(
    colorbar.ax.get_yticklabels(),
    color=TEXT
)
colorbar.set_label(
    "Pearson Correlation",
    color=TEXT
)

plt.tight_layout()

plt.savefig(
    figure_folder / "airport_correlation_heatmap.png",
    dpi=300,
    facecolor=BG,
    bbox_inches="tight"
)

plt.close()

print("\n4. Generated Files")
print("airport_correlation_matrix.csv")
print("airport_metric_relationships.csv")
print("airport_correlation_heatmap.png")

print("\nCorrelation interpretation:")
print(
    "Correlation measures linear association, "
    "not causation. Long-term trends and seasonality "
    "may influence these results."
)

print("\nHeatmap saved in:", figure_folder)

print("\n========== CORRELATION ANALYSIS COMPLETED ==========")
