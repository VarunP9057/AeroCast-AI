
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DIR = ROOT / "reports/aircraft_forecasting"
EXP2_DIR = ROOT / "reports/aircraft_forecasting_experiment2"
EXP3_DIR = ROOT / "reports/aircraft_forecasting_experiment3"

OUTPUT_DIR = ROOT / "reports/aircraft_forecasting_final"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# 1. LOAD EXISTING MODEL EVALUATION REPORTS
# --------------------------------------------------

original_metrics = read_csv(
    ORIGINAL_DIR / "aircraft_model_comparison.csv"
)

experiment2_metrics = read_csv(
    EXP2_DIR / "aircraft_experiment2_model_comparison.csv"
)

experiment3_metrics = read_csv(
    EXP3_DIR / "aircraft_experiment3_metrics.csv"
)

original_predictions = read_csv(
    ORIGINAL_DIR / "aircraft_model_predictions.csv"
)

experiment2_predictions = read_csv(
    EXP2_DIR / "aircraft_experiment2_predictions.csv"
)

experiment3_predictions = read_csv(
    EXP3_DIR / "aircraft_experiment3_predictions.csv"
)

original_history = read_csv(
    ORIGINAL_DIR / "aircraft_ann_training_history.csv"
)

experiment2_history = read_csv(
    EXP2_DIR / "aircraft_experiment2_training_history.csv"
)

experiment3_history = read_csv(
    EXP3_DIR / "aircraft_experiment3_training_history.csv"
)

walk_forward = read_csv(
    EXP3_DIR / "walk_forward_validation.csv"
)


# --------------------------------------------------
# 2. CONSOLIDATE MODEL PERFORMANCE
# --------------------------------------------------

original_metrics["Model"] = original_metrics["Model"].replace({
    "ANN": "Original ANN",
    "Linear Regression": "Linear Regression (Original Features)",
    "Random Forest": "Random Forest (Original Features)"
})

experiment2_row = experiment2_metrics[
    experiment2_metrics["Model"] == "ANN_Experiment_2"
].copy()

experiment2_row["Model"] = "ANN Experiment 2"

experiment3_metrics["Model"] = "ANN Experiment 3"

final_metrics = pd.concat(
    [
        original_metrics,
        experiment2_row,
        experiment3_metrics
    ],
    ignore_index=True
)

final_metrics = final_metrics[
    ["Model", "MAE", "RMSE", "R2", "MAPE", "WAPE"]
]

final_metrics = final_metrics.sort_values(
    "MAE"
).reset_index(drop=True)

final_metrics.insert(
    0,
    "Rank",
    range(1, len(final_metrics) + 1)
)

final_metrics.to_csv(
    OUTPUT_DIR / "aircraft_final_model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 3. COMBINE ANN PREDICTIONS
# --------------------------------------------------


# --------------------------------------------------
# 3. COMBINE ANN PREDICTIONS
# --------------------------------------------------
# Standardize all dates to YYYY-MM

# --------------------------------------------------
# 3. COMBINE ANN PREDICTIONS
# --------------------------------------------------

# Standardize all dates to YYYY-MM
for df in [
    original_predictions,
    experiment2_predictions,
    experiment3_predictions
]:
    df["YearMonth"] = pd.to_datetime(
        df["YearMonth"].astype(str).str.strip(),
        errors="raise"
    ).dt.to_period("M").astype(str)

original_ann = original_predictions[
    ["YearMonth", "Actual", "ANN"]
].rename(columns={"ANN": "Original_ANN"})

experiment2_ann = experiment2_predictions[
    ["YearMonth", "Actual", "ANN_Experiment_2"]
]

experiment3_ann = experiment3_predictions[
    ["YearMonth", "Actual", "Experiment3_ANN"]
].rename(columns={"Experiment3_ANN": "ANN_Experiment_3"})

# Check that all three reports contain the same test months
common_months = (
    set(original_ann["YearMonth"])
    & set(experiment2_ann["YearMonth"])
    & set(experiment3_ann["YearMonth"])
)

if not common_months:
    raise ValueError("No common test months found between the ANN reports.")

# Merge predictions using only common months
original_ann = original_ann[
    original_ann["YearMonth"].isin(common_months)
]

experiment2_ann = experiment2_ann[
    experiment2_ann["YearMonth"].isin(common_months)
]

experiment3_ann = experiment3_ann[
    experiment3_ann["YearMonth"].isin(common_months)
]

actual_check = original_ann[
    ["YearMonth", "Actual"]
].merge(
    experiment2_ann[["YearMonth", "Actual"]],
    on="YearMonth",
    suffixes=("_original", "_exp2"),
    validate="one_to_one"
).merge(
    experiment3_ann[["YearMonth", "Actual"]],
    on="YearMonth",
    validate="one_to_one"
)

if not (
    np.allclose(actual_check["Actual_original"], actual_check["Actual_exp2"])
    and np.allclose(actual_check["Actual_original"], actual_check["Actual"])
):
    raise ValueError("Actual test values differ between experiments.")

combined = original_ann.merge(
    experiment2_ann[["YearMonth", "ANN_Experiment_2"]],
    on="YearMonth",
    how="inner",
    validate="one_to_one"
)

combined = combined.merge(
    experiment3_ann[["YearMonth", "ANN_Experiment_3"]],
    on="YearMonth",
    how="inner",
    validate="one_to_one"
)

combined = combined.sort_values("YearMonth").reset_index(drop=True)

if combined.empty:
    raise ValueError("Combined ANN predictions are empty.")

combined.to_csv(
    OUTPUT_DIR / "aircraft_combined_ann_predictions.csv",
    index=False
)

print("\nANN prediction comparison:")
print("Common test months:", len(combined))
print("Test period:", combined["YearMonth"].iloc[0], "to", combined["YearMonth"].iloc[-1])


# --------------------------------------------------
# 4. RESIDUAL ANALYSIS
# --------------------------------------------------

for model in [
    "Original_ANN",
    "ANN_Experiment_2",
    "ANN_Experiment_3"
]:
    combined[f"{model}_Residual"] = (
        combined["Actual"] - combined[model]
    )

    combined[f"{model}_Absolute_Error"] = (
        combined[f"{model}_Residual"].abs()
    )

residual_columns = [
    "YearMonth",
    "Actual",
    "Original_ANN_Residual",
    "ANN_Experiment_2_Residual",
    "ANN_Experiment_3_Residual",
    "Original_ANN_Absolute_Error",
    "ANN_Experiment_2_Absolute_Error",
    "ANN_Experiment_3_Absolute_Error"
]

combined[residual_columns].to_csv(
    OUTPUT_DIR / "aircraft_ann_residual_analysis.csv",
    index=False
)


# --------------------------------------------------
# 5. ACTUAL VS PREDICTED ANN COMPARISON
# --------------------------------------------------

plt.figure(figsize=(15, 7))

plt.plot(
    combined["YearMonth"],
    combined["Actual"],
    label="Actual",
    linewidth=2.5
)

plt.plot(
    combined["YearMonth"],
    combined["Original_ANN"],
    label="Original ANN",
    linewidth=1.8
)

plt.plot(
    combined["YearMonth"],
    combined["ANN_Experiment_2"],
    label="ANN Experiment 2",
    linewidth=1.8
)

plt.plot(
    combined["YearMonth"],
    combined["ANN_Experiment_3"],
    label="ANN Experiment 3",
    linewidth=1.8
)

plt.title("Aircraft Movement: Actual vs ANN Predictions")
plt.xlabel("Test Month")
plt.ylabel("Aircraft Landing Count")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)

save_chart("01_ann_actual_vs_predicted.png")


# --------------------------------------------------
# 6. MAE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    final_metrics["Model"],
    final_metrics["MAE"]
)

plt.title("Aircraft Forecasting - MAE Comparison")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)

save_chart("02_mae_comparison.png")


# --------------------------------------------------
# 7. RMSE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    final_metrics["Model"],
    final_metrics["RMSE"]
)

plt.title("Aircraft Forecasting - RMSE Comparison")
plt.ylabel("Root Mean Squared Error")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)

save_chart("03_rmse_comparison.png")


# --------------------------------------------------
# 8. R2 COMPARISON
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    final_metrics["Model"],
    final_metrics["R2"]
)

plt.title("Aircraft Forecasting - R² Comparison")
plt.ylabel("R² Score")
plt.xticks(rotation=30, ha="right")
plt.axhline(0, linewidth=1)
plt.grid(axis="y", alpha=0.3)

save_chart("04_r2_comparison.png")


# --------------------------------------------------
# 9. ORIGINAL ANN RESIDUAL DISTRIBUTION
# --------------------------------------------------

original_residuals = (
    combined["Actual"] - combined["Original_ANN"]
)

plt.figure(figsize=(10, 6))

plt.hist(
    original_residuals,
    bins=20,
    edgecolor="black"
)

plt.axvline(0, linestyle="--", linewidth=2)

plt.title("Original ANN - Residual Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)

save_chart("05_original_ann_residual_distribution.png")


# --------------------------------------------------
# 10. ORIGINAL ANN RESIDUALS OVER TIME
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    combined["YearMonth"],
    original_residuals,
    marker="o",
    linewidth=1.5
)

plt.axhline(0, linestyle="--")

plt.title("Original ANN - Residuals Over Time")
plt.xlabel("Test Month")
plt.ylabel("Residual")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

save_chart("06_original_ann_residuals_over_time.png")


# --------------------------------------------------
# 11. TRAINING HISTORY COMPARISON
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

histories = [
    ("Original ANN", original_history),
    ("ANN Experiment 2", experiment2_history),
    ("ANN Experiment 3", experiment3_history)
]

for ax, (name, history) in zip(axes, histories):

    ax.plot(
        history["loss"],
        label="Training Loss"
    )

    ax.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle("Aircraft ANN Training and Validation Loss")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_ann_training_history_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 12. WALK-FORWARD VALIDATION REPORT
# --------------------------------------------------

walk_forward.to_csv(
    OUTPUT_DIR / "aircraft_walk_forward_validation.csv",
    index=False
)


# --------------------------------------------------
# 13. FINAL SUMMARY
# --------------------------------------------------

best_model = final_metrics.iloc[0]

summary = f"""
AEROCAST AI
AIRCRAFT FORECASTING - FINAL EVALUATION

Common test records: {len(combined)}
Test period: {combined['YearMonth'].iloc[0]} to {combined['YearMonth'].iloc[-1]}

Best model by MAE:
{best_model['Model']}

MAE: {best_model['MAE']:.2f}
RMSE: {best_model['RMSE']:.2f}
R2: {best_model['R2']:.4f}
MAPE: {best_model['MAPE']:.2f}%
WAPE: {best_model['WAPE']:.2f}%

Original ANN remains the selected champion from the
previous aircraft model evaluation.

No model was retrained or overwritten by this script.
"""

(OUTPUT_DIR / "aircraft_final_evaluation_summary.txt").write_text(
    summary,
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("AIRCRAFT FORECASTING FINAL EVALUATION COMPLETED")
print("=" * 65)

print("\nMODEL COMPARISON:")
print(final_metrics.to_string(index=False))

print("\nCOMMON TEST RECORDS:", len(combined))

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nGENERATED CHARTS:")
for file in sorted(FIGURE_DIR.glob("*.png")):
    print("-", file.name)

print("\nEVALUATION REPORTS:")
for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print("-", file.name)

print("\nBest model by MAE:", best_model["Model"])
print("MAE:", round(best_model["MAE"], 2))
print("\nEvaluation completed successfully.")
