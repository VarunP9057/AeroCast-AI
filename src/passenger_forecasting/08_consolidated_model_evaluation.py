
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data/features/passenger_features.csv"

EVALUATION_DIR = ROOT / "reports/evaluation"
EXPERIMENT_DIR = ROOT / "reports/ann_experiment"

OUTPUT_DIR = ROOT / "reports/consolidated_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Passenger Count"

print("\n==============================================")
print("AEROCAST AI - CONSOLIDATED MODEL EVALUATION")
print("==============================================")


# =====================================================
# 2. LOAD TEST DATA
# =====================================================

df = pd.read_csv(FEATURE_PATH)

df = df.sort_values("YearMonth").reset_index(drop=True)

X = df.drop(columns=[TARGET, "YearMonth"])
y = pd.to_numeric(df[TARGET], errors="coerce")

X = X.apply(pd.to_numeric, errors="coerce")

valid = X.notna().all(axis=1) & y.notna()

df = df.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

split = int(len(df) * 0.80)

test_df = df.iloc[split:].reset_index(drop=True)

test_dates = test_df["YearMonth"].to_numpy()
actual_values = y.iloc[split:].to_numpy()

print("\nTEST DATA INFORMATION")
print("Total valid records:", len(df))
print("Test records:", len(actual_values))
print("Test start:", test_dates[0])
print("Test end:", test_dates[-1])


# =====================================================
# 3. HELPER FUNCTION TO LOAD PREDICTIONS
# =====================================================

def load_prediction_file(file_path, model_name):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    actual_column = next(
        (
            col for col in data.columns
            if "actual" in col.lower()
        ),
        None
    )

    predicted_column = next(
        (
            col for col in data.columns
            if "predicted" in col.lower()
        ),
        None
    )

    if actual_column is None or predicted_column is None:
        raise ValueError(
            f"Could not identify actual/predicted columns "
            f"in {file_path.name}. Columns: {list(data.columns)}"
        )

    actual = pd.to_numeric(
        data[actual_column],
        errors="coerce"
    ).to_numpy()

    predicted = pd.to_numeric(
        data[predicted_column],
        errors="coerce"
    ).to_numpy()

    if len(actual) != len(actual_values):
        raise ValueError(
            f"{model_name}: Expected {len(actual_values)} "
            f"predictions, found {len(actual)}."
        )

    if len(predicted) != len(actual_values):
        raise ValueError(
            f"{model_name}: Prediction count does not match test data."
        )

    if not np.isfinite(actual).all():
        raise ValueError(f"{model_name}: Invalid actual values.")

    if not np.isfinite(predicted).all():
        raise ValueError(f"{model_name}: Invalid predictions.")

    if not np.allclose(actual, actual_values, rtol=0, atol=1):
        raise ValueError(
            f"{model_name}: Actual passenger values do not match "
            "the current chronological test split."
        )

    return np.maximum(predicted, 0)


# =====================================================
# 4. LOAD ALL FOUR MODEL PREDICTIONS
# =====================================================

predictions = {}

predictions["Linear Regression"] = load_prediction_file(
    EVALUATION_DIR / "linear_regression_predictions.csv",
    "Linear Regression"
)

predictions["Random Forest"] = load_prediction_file(
    EVALUATION_DIR / "random_forest_predictions.csv",
    "Random Forest"
)

predictions["Original ANN"] = load_prediction_file(
    EVALUATION_DIR / "ann_predictions.csv",
    "Original ANN"
)

experiment_file = (
    EXPERIMENT_DIR / "ann_experiment_predictions.csv"
)

if not experiment_file.exists():
    raise FileNotFoundError(experiment_file)

experiment_df = pd.read_csv(experiment_file)

if "ANN Predicted Passenger Count" not in experiment_df.columns:
    raise ValueError(
        "Experimental ANN prediction column not found."
    )

experiment_actual = experiment_df[
    "Actual Passenger Count"
].to_numpy()

experiment_prediction = experiment_df[
    "ANN Predicted Passenger Count"
].to_numpy()

if len(experiment_prediction) != len(actual_values):
    raise ValueError("Experimental ANN prediction count mismatch.")

if not np.allclose(
    experiment_actual,
    actual_values,
    rtol=0,
    atol=1
):
    raise ValueError(
        "Experimental ANN actual values do not match test data."
    )

predictions["Experimental ANN"] = np.maximum(
    experiment_prediction,
    0
)

print("\nPREDICTION FILES VERIFIED")
for name, values in predictions.items():
    print(f"{name}: {len(values)} predictions verified")


# =====================================================
# 5. MODEL EVALUATION
# =====================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(np.abs(actual))
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
        "MAPE (%)": mape,
        "WAPE (%)": wape
    }


# =====================================================
# 6. CREATE COMPARISON TABLE
# =====================================================

results = []

for model_name, predicted in predictions.items():

    metrics = calculate_metrics(
        actual_values,
        predicted
    )

    results.append({
        "Model": model_name,
        **metrics
    })

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "MAE"
).reset_index(drop=True)

comparison.insert(
    0,
    "Rank",
    range(1, len(comparison) + 1)
)

comparison.to_csv(
    OUTPUT_DIR / "consolidated_model_comparison.csv",
    index=False
)


# =====================================================
# 7. SAVE ALL MODEL PREDICTIONS
# =====================================================

prediction_table = pd.DataFrame({
    "YearMonth": test_dates,
    "Actual Passenger Count": actual_values
})

for model_name, predicted in predictions.items():

    safe_name = model_name.lower().replace(" ", "_")

    prediction_table[
        f"{safe_name}_prediction"
    ] = predicted

    prediction_table[
        f"{safe_name}_absolute_error"
    ] = np.abs(actual_values - predicted)

prediction_table.to_csv(
    OUTPUT_DIR / "consolidated_model_predictions.csv",
    index=False
)


# =====================================================
# 8. MODEL COMPARISON CHART
# =====================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.title("AeroCast AI - Model Comparison by MAE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error (Passengers)")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison_mae.png",
    dpi=300
)

plt.close()


# =====================================================
# 9. RMSE COMPARISON CHART
# =====================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.title("AeroCast AI - Model Comparison by RMSE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Root Mean Squared Error (Passengers)")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison_rmse.png",
    dpi=300
)

plt.close()


# =====================================================
# 10. R2 COMPARISON CHART
# =====================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison["Model"],
    comparison["R2 Score"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.title("AeroCast AI - Model Comparison by R2 Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison_r2.png",
    dpi=300
)

plt.close()


# =====================================================
# 11. ACTUAL VS ALL MODEL PREDICTIONS
# =====================================================

plt.figure(figsize=(15, 7))

plt.plot(
    test_dates,
    actual_values,
    label="Actual Passenger Traffic",
    linewidth=3
)

for model_name, predicted in predictions.items():

    plt.plot(
        test_dates,
        predicted,
        label=model_name,
        linewidth=1.5
    )

plt.title(
    "AeroCast AI - Actual vs Predicted Passenger Traffic"
)

plt.xlabel("Year and Month")
plt.ylabel("Passenger Count")

plt.xticks(
    rotation=45
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "all_models_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# =====================================================
# 12. FINAL BEST MODEL
# =====================================================

best_model = comparison.iloc[0]

print("\n==============================================")
print("CONSOLIDATED MODEL EVALUATION RESULTS")
print("==============================================")

print(
    comparison.to_string(
        index=False,
        formatters={
            "MAE": "{:,.2f}".format,
            "RMSE": "{:,.2f}".format,
            "R2 Score": "{:.4f}".format,
            "MAPE (%)": "{:.2f}".format,
            "WAPE (%)": "{:.2f}".format
        }
    )
)

print("\n==============================================")
print("BEST MODEL BASED ON MAE")
print("==============================================")

print("Model:", best_model["Model"])
print("MAE:", round(best_model["MAE"], 2))
print("RMSE:", round(best_model["RMSE"], 2))
print("R2 Score:", round(best_model["R2 Score"], 4))
print("MAPE:", round(best_model["MAPE (%)"], 2), "%")
print("WAPE:", round(best_model["WAPE (%)"], 2), "%")


# =====================================================
# 13. SAVE FINAL SUMMARY
# =====================================================

summary = f"""
AEROCAST AI
CONSOLIDATED PASSENGER FORECASTING MODEL EVALUATION

Test period: {test_dates[0]} to {test_dates[-1]}
Test records: {len(actual_values)}

Models evaluated:
1. Linear Regression
2. Random Forest
3. Original ANN
4. Experimental ANN

Best model based on MAE: {best_model["Model"]}

MAE: {best_model["MAE"]:.2f}
RMSE: {best_model["RMSE"]:.2f}
R2 Score: {best_model["R2 Score"]:.4f}
MAPE: {best_model["MAPE (%)"]:.2f}%
WAPE: {best_model["WAPE (%)"]:.2f}%

Evaluation method:
Chronological 80:20 train-test split.
All four models are compared on the same test records.
"""

with open(
    OUTPUT_DIR / "final_model_evaluation_summary.txt",
    "w"
) as file:
    file.write(summary)


# =====================================================
# 14. DISPLAY OUTPUT FILES
# =====================================================

print("\nSAVED OUTPUT FILES")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\n==============================================")
print("CONSOLIDATED EVALUATION COMPLETED")
print("==============================================")
 