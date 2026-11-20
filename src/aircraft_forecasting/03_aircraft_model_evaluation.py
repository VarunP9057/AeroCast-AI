
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILE = PROJECT_ROOT / "reports" / "aircraft_forecasting" / "aircraft_model_predictions.csv"
COMPARISON_FILE = PROJECT_ROOT / "reports" / "aircraft_forecasting" / "aircraft_model_comparison.csv"
HISTORY_FILE = PROJECT_ROOT / "reports" / "aircraft_forecasting" / "aircraft_ann_training_history.csv"

OUTPUT_FOLDER = PROJECT_ROOT / "reports" / "aircraft_forecasting" / "figures"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

predictions = pd.read_csv(PREDICTION_FILE)
comparison = pd.read_csv(COMPARISON_FILE)
history = pd.read_csv(HISTORY_FILE)

predictions["YearMonth"] = pd.to_datetime(predictions["YearMonth"])

predictions["ANN_Residual"] = (
    predictions["Actual"] - predictions["ANN"]
)

predictions["ANN_Absolute_Error"] = (
    predictions["ANN_Residual"].abs()
)

plt.rcParams.update({
    "figure.figsize": (13, 6),
    "axes.facecolor": "#F1F3F1",
    "figure.facecolor": "#F1F3F1",
    "axes.edgecolor": "#89938C",
    "axes.labelcolor": "#29332D",
    "xtick.color": "#29332D",
    "ytick.color": "#29332D",
    "text.color": "#29332D",
    "font.size": 10
})

# 1. Actual vs Predicted - All Models

plt.figure()

plt.plot(
    predictions["YearMonth"],
    predictions["Actual"],
    label="Actual",
    color="#29332D",
    linewidth=2.5
)

for model, color in [
    ("ANN", "#A1C83C"),
    ("Linear_Regression", "#647BFF"),
    ("Random_Forest", "#E69B42")
]:
    plt.plot(
        predictions["YearMonth"],
        predictions[model],
        label=model.replace("_", " "),
        linewidth=1.8,
        color=color
    )

plt.title("Aircraft Landings: Actual vs Model Predictions")
plt.xlabel("Year")
plt.ylabel("Monthly Landing Count")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "01_all_models_predictions.png", dpi=300)
plt.close()

# 2. ANN Actual vs Predicted

plt.figure()

plt.plot(
    predictions["YearMonth"],
    predictions["Actual"],
    label="Actual",
    color="#29332D",
    linewidth=2.5
)

plt.plot(
    predictions["YearMonth"],
    predictions["ANN"],
    label="ANN Prediction",
    color="#A1C83C",
    linewidth=2
)

plt.title("ANN: Actual vs Predicted Aircraft Landings")
plt.xlabel("Year")
plt.ylabel("Monthly Landing Count")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "02_ann_actual_vs_predicted.png", dpi=300)
plt.close()

# 3. ANN Residual Distribution

plt.figure()

plt.hist(
    predictions["ANN_Residual"],
    bins=20,
    edgecolor="#29332D",
    alpha=0.85
)

plt.axvline(0, linestyle="--", color="#D9534F")

plt.title("ANN Prediction Error Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "03_ann_residual_distribution.png", dpi=300)
plt.close()

# 4. ANN Residuals Over Time

plt.figure()

plt.axhline(0, linestyle="--", color="#D9534F")

plt.plot(
    predictions["YearMonth"],
    predictions["ANN_Residual"],
    marker="o",
    markersize=3,
    linewidth=1.5,
    color="#647BFF"
)

plt.title("ANN Prediction Residuals Over Time")
plt.xlabel("Year")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "04_ann_residuals_over_time.png", dpi=300)
plt.close()

# 5. Model MAE Comparison

plt.figure()

plt.bar(
    comparison["Model"],
    comparison["MAE"],
    color=["#A1C83C", "#647BFF", "#E69B42", "#89938C"]
)

plt.title("Model Comparison - MAE")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "05_model_mae_comparison.png", dpi=300)
plt.close()

# 6. Model RMSE Comparison

plt.figure()

plt.bar(
    comparison["Model"],
    comparison["RMSE"],
    color=["#A1C83C", "#647BFF", "#E69B42", "#89938C"]
)

plt.title("Model Comparison - RMSE")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "06_model_rmse_comparison.png", dpi=300)
plt.close()

# 7. Model R2 Comparison

plt.figure()

plt.bar(
    comparison["Model"],
    comparison["R2"],
    color=["#A1C83C", "#647BFF", "#E69B42", "#89938C"]
)

plt.axhline(0, linestyle="--", color="#D9534F")

plt.title("Model Comparison - R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "07_model_r2_comparison.png", dpi=300)
plt.close()

# 8. ANN Training and Validation Loss

plt.figure()

plt.plot(
    history["loss"],
    label="Training Loss",
    color="#647BFF",
    linewidth=2
)

plt.plot(
    history["val_loss"],
    label="Validation Loss",
    color="#A1C83C",
    linewidth=2
)

plt.title("ANN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "08_ann_training_history.png", dpi=300)
plt.close()

# 9. Largest ANN Prediction Errors

largest_errors = predictions.nlargest(
    10, "ANN_Absolute_Error"
)

plt.figure(figsize=(12, 6))

plt.bar(
    largest_errors["YearMonth"].dt.strftime("%Y-%m"),
    largest_errors["ANN_Absolute_Error"],
    color="#A1C83C"
)

plt.title("Top 10 Largest ANN Prediction Errors")
plt.xlabel("Month")
plt.ylabel("Absolute Prediction Error")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "09_ann_largest_errors.png", dpi=300)
plt.close()

# Save residual analysis

predictions.to_csv(
    PROJECT_ROOT / "reports" / "aircraft_forecasting" / "aircraft_residual_analysis.csv",
    index=False
)

print("\n========== AEROCAST AI ==========")
print("AIRCRAFT MODEL EVALUATION")
print("================================")

print("\nANN Mean Residual:", round(predictions["ANN_Residual"].mean(), 2))
print("ANN Mean Absolute Error:", round(predictions["ANN_Absolute_Error"].mean(), 2))
print("ANN Maximum Absolute Error:", round(predictions["ANN_Absolute_Error"].max(), 2))

print("\nLargest 5 ANN Errors:")
print(
    largest_errors[
        ["YearMonth", "Actual", "ANN", "ANN_Absolute_Error"]
    ].to_string(index=False)
)

print("\nGenerated 9 evaluation charts:")
for file in sorted(OUTPUT_FOLDER.glob("*.png")):
    print(file.name)

print("\nResidual report saved:")
print(PROJECT_ROOT / "reports" / "aircraft_forecasting" / "aircraft_residual_analysis.csv")

print("\n========== EVALUATION COMPLETED ==========")
