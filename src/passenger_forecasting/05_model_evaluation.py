
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).resolve().parents[2]

report_folder = project_path / "reports" / "evaluation"
figure_folder = project_path / "reports" / "figures"

figure_folder.mkdir(parents=True, exist_ok=True)

NAVY = "#101B2D"
CYAN = "#00CFE8"

print("\n========== AEROCAST AI ==========")
print("MODEL EVALUATION AND COMPARISON")
print("================================")

# Load model evaluation results
files = [
    "linear_regression_metrics.csv",
    "random_forest_metrics.csv",
    "ann_metrics.csv"
]

all_metrics = []

for file in files:
    path = report_folder / file

    if not path.exists():
        raise FileNotFoundError(
            f"{file} not found. Run the corresponding model first."
        )

    metrics = pd.read_csv(path)
    all_metrics.append(metrics)

comparison = pd.concat(all_metrics, ignore_index=True)

comparison = comparison.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

# Save comparison table
comparison.to_csv(
    report_folder / "model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

best_model = comparison.iloc[0]

print("\nBest Model Based on MAE:", best_model["Model"])
print("Lowest MAE:", round(best_model["MAE"], 2))

# Visualization settings
plt.rcParams["figure.facecolor"] = NAVY
plt.rcParams["axes.facecolor"] = NAVY
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"

# 1. MAE Comparison
plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["MAE"],
    color=CYAN,
    width=0.55
)

plt.title("Model Comparison - MAE", color=CYAN, fontsize=15)
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    figure_folder / "model_mae_comparison.png",
    dpi=300,
    facecolor=NAVY
)

plt.close()

# 2. RMSE Comparison
plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["RMSE"],
    color=CYAN,
    width=0.55
)

plt.title("Model Comparison - RMSE", color=CYAN, fontsize=15)
plt.xlabel("Machine Learning Models")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    figure_folder / "model_rmse_comparison.png",
    dpi=300,
    facecolor=NAVY
)

plt.close()

# 3. R2 Score Comparison
plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["R2 Score"],
    color=CYAN,
    width=0.55
)

plt.axhline(0, color="white", linewidth=1)

plt.title("Model Comparison - R2 Score", color=CYAN, fontsize=15)
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

plt.savefig(
    figure_folder / "model_r2_comparison.png",
    dpi=300,
    facecolor=NAVY
)

plt.close()

print("\nGenerated Visualizations:")
print("1. model_mae_comparison.png")
print("2. model_rmse_comparison.png")
print("3. model_r2_comparison.png")

print("\nComparison CSV:")
print("model_comparison.csv")

print("\n========== EVALUATION COMPLETED ==========")
