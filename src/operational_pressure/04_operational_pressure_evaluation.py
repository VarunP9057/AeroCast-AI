
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path.cwd()

REPORT_DIR = ROOT / "reports/operational_pressure"
FIGURE_DIR = REPORT_DIR / "figures"

PREDICTION_PATH = REPORT_DIR / "operational_pressure_model_predictions.csv"
COMPARISON_PATH = REPORT_DIR / "operational_pressure_model_comparison.csv"
HISTORY_PATH = REPORT_DIR / "operational_pressure_ann_training_history.csv"
ANN_VALIDATION_PATH = REPORT_DIR / "operational_pressure_ann_walk_forward.csv"
ANN_SUMMARY_PATH = REPORT_DIR / "operational_pressure_ann_validation_summary.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print("Saved:", filename)


# --------------------------------------------------
# LOAD REPORTS
# --------------------------------------------------

predictions = pd.read_csv(PREDICTION_PATH)
comparison = pd.read_csv(COMPARISON_PATH)
history = pd.read_csv(HISTORY_PATH)
ann_validation = pd.read_csv(ANN_VALIDATION_PATH)

predictions["YearMonth"] = pd.to_datetime(predictions["YearMonth"])

actual_column = "Actual"
ann_prediction_column = "ANN"

if actual_column not in predictions.columns:
    raise ValueError(
        f"Actual column not found. Available columns: "
        f"{predictions.columns.tolist()}"
    )

if ann_prediction_column not in predictions.columns:
    raise ValueError(
        f"ANN column not found. Available columns: "
        f"{predictions.columns.tolist()}"
    )

print("=" * 65)
print("OPERATIONAL PRESSURE MODEL EVALUATION")
print("=" * 65)

print("\nPrediction columns:", predictions.columns.tolist())
print("Comparison columns:", comparison.columns.tolist())
print("History columns:", history.columns.tolist())


# --------------------------------------------------
# 1. ACTUAL VS PREDICTED PRESSURE
# --------------------------------------------------

plt.figure(figsize=(15, 6))

plt.plot(
    predictions["YearMonth"],
    predictions[actual_column],
    label="Actual Pressure",
    linewidth=2.5
)

plt.plot(
    predictions["YearMonth"],
    predictions[ann_prediction_column],
    label="ANN Predicted Pressure",
    linewidth=2,
    linestyle="--"
)

plt.title(
    "Operational Pressure: Actual vs ANN Predicted",
    fontsize=16
)
plt.xlabel("Year")
plt.ylabel("Operational Pressure Index")
plt.legend()
plt.xticks(rotation=45)

save_figure("01_actual_vs_predicted_pressure.png")


# --------------------------------------------------
# 2. MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

metric_columns = ["MAE", "RMSE", "R2"]

for metric in metric_columns:
    if metric not in comparison.columns:
        raise ValueError(
            f"Missing {metric} in comparison report."
        )

if "Model" not in comparison.columns:
    raise ValueError("Model column not found in comparison report.")

for metric in metric_columns:

    plt.figure(figsize=(12, 6))

    ordered = comparison.sort_values(
        metric,
        ascending=(metric != "R2")
    )

    sns.barplot(
        data=ordered,
        x="Model",
        y=metric
    )

    plt.title(
        f"Operational Pressure Model Comparison: {metric}",
        fontsize=15
    )

    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")

    save_figure(
        f"02_model_comparison_{metric.lower()}.png"
    )


# --------------------------------------------------
# 3. RESIDUAL ERROR DISTRIBUTION
# --------------------------------------------------

predictions["ANN_Residual"] = (
    predictions[actual_column]
    - predictions[ann_prediction_column]
)

plt.figure(figsize=(10, 6))

sns.histplot(
    predictions["ANN_Residual"],
    kde=True,
    bins=20
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2
)

plt.title(
    "ANN Residual Error Distribution",
    fontsize=15
)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")

save_figure("03_ann_residual_distribution.png")


# --------------------------------------------------
# 4. RESIDUALS OVER TIME
# --------------------------------------------------

plt.figure(figsize=(15, 6))

plt.plot(
    predictions["YearMonth"],
    predictions["ANN_Residual"],
    linewidth=1.8,
    marker="o",
    markersize=3
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2
)

plt.title(
    "ANN Residuals Over Time",
    fontsize=15
)

plt.xlabel("Year")
plt.ylabel("Residual Error")
plt.xticks(rotation=45)

save_figure("04_ann_residuals_over_time.png")


# --------------------------------------------------
# 5. ANN TRAINING HISTORY
# --------------------------------------------------

history_columns = history.columns.tolist()

loss_column = next(
    (
        col for col in history_columns
        if col.lower() == "loss"
    ),
    None
)

val_loss_column = next(
    (
        col for col in history_columns
        if col.lower() == "val_loss"
    ),
    None
)

if loss_column and val_loss_column:

    plt.figure(figsize=(11, 6))

    plt.plot(
        history[loss_column],
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history[val_loss_column],
        label="Validation Loss",
        linewidth=2
    )

    plt.title(
        "Operational Pressure ANN Training History",
        fontsize=15
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    save_figure("05_ann_training_history.png")

else:
    print(
        "\nTraining history loss columns not found. "
        "Skipping chart 5."
    )


# --------------------------------------------------
# 6. ANN WALK-FORWARD VALIDATION
# --------------------------------------------------

plt.figure(figsize=(11, 6))

sns.barplot(
    data=ann_validation,
    x="Fold",
    y="MAE"
)

plt.title(
    "ANN Walk-Forward Validation: MAE by Fold",
    fontsize=15
)

plt.xlabel("Validation Fold")
plt.ylabel("MAE")

save_figure("06_ann_walk_forward_validation.png")



# --------------------------------------------------
# 7. WALK-FORWARD VALIDATION MODEL COMPARISON
# --------------------------------------------------

linear_cv_path = (
    REPORT_DIR / "operational_pressure_cross_validation.csv"
)

if linear_cv_path.exists():

    traditional_cv = pd.read_csv(linear_cv_path)

    traditional_cv = traditional_cv.rename(columns={
        "Average_MAE": "MAE",
        "Average_RMSE": "RMSE",
        "Average_R2": "R2"
    })

    ann_cv = pd.DataFrame([{
        "Model": "ANN",
        "MAE": ann_validation["MAE"].mean(),
        "RMSE": ann_validation["RMSE"].mean(),
        "R2": ann_validation["R2"].mean()
    }])

    cv_comparison = pd.concat(
        [
            traditional_cv[["Model", "MAE", "RMSE", "R2"]],
            ann_cv
        ],
        ignore_index=True
    )

    cv_comparison.to_csv(
        REPORT_DIR / "operational_pressure_combined_cv_comparison.csv",
        index=False
    )

    for metric in ["MAE", "RMSE", "R2"]:

        plt.figure(figsize=(12, 6))

        ordered = cv_comparison.sort_values(
            metric,
            ascending=(metric != "R2")
        )

        sns.barplot(
            data=ordered,
            x="Model",
            y=metric
        )

        plt.title(
            f"Walk-Forward Validation Model Comparison: {metric}",
            fontsize=15
        )

        plt.xlabel("Model")
        plt.ylabel(f"Average Validation {metric}")
        plt.xticks(rotation=30, ha="right")

        save_figure(
            f"07_walk_forward_comparison_{metric.lower()}.png"
        )

    print("\nCombined Walk-Forward Validation Results:")
    print(cv_comparison.round(4).to_string(index=False))

else:
    print("\nTraditional cross-validation report not found.")

# --------------------------------------------------
# FINAL TEST METRICS FOR ALL MODELS
# --------------------------------------------------

print("\n" + "=" * 65)
print("HELD-OUT TEST PERFORMANCE")
print("=" * 65)

test_results = []

for model_name in [
    col for col in predictions.columns
    if col not in ["YearMonth", "Actual", "ANN_Residual"]
]:

    y_true = predictions[actual_column]
    y_pred = predictions[model_name]

    test_results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred)
    })

test_results_df = pd.DataFrame(test_results)

print(test_results_df.round(4).to_string(index=False))

test_results_df.to_csv(
    REPORT_DIR / "operational_pressure_final_test_metrics.csv",
    index=False
)


# --------------------------------------------------
# ANN WALK-FORWARD SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("ANN WALK-FORWARD VALIDATION SUMMARY")
print("=" * 65)

print("Average MAE:", round(ann_validation["MAE"].mean(), 4))
print("Average RMSE:", round(ann_validation["RMSE"].mean(), 4))
print("Average R2:", round(ann_validation["R2"].mean(), 4))

if ANN_SUMMARY_PATH.exists():

    summary = pd.read_csv(ANN_SUMMARY_PATH)

    print("\nSaved validation summary:")
    print(summary.to_string(index=False))


# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Total visualization files:", len(list(FIGURE_DIR.glob("*.png"))))
print("Figure folder:", FIGURE_DIR)
print("Final metrics saved in:", REPORT_DIR)
