
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/operational_pressure_features.csv"
SOURCE_MODEL_DIR = ROOT / "models/operational_pressure"
PREDICTION_PATH = ROOT / "reports/operational_pressure/operational_pressure_model_predictions.csv"

FINAL_MODEL_DIR = ROOT / "models/operational_pressure_final"
REPORT_DIR = ROOT / "reports/operational_pressure"

FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Operational Pressure Index"
TEST_START = "2021-06-01"

print("=" * 65)
print("OPERATIONAL PRESSURE FINAL MODEL PREPARATION")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD DATA AND EXISTING MODEL
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

model_path = SOURCE_MODEL_DIR / "linear_regression.pkl"
feature_path = SOURCE_MODEL_DIR / "operational_pressure_feature_names.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not feature_path.exists():
    raise FileNotFoundError(f"Feature file not found: {feature_path}")

model = joblib.load(model_path)
feature_names = joblib.load(feature_path)

if isinstance(feature_names, np.ndarray):
    feature_names = feature_names.tolist()

feature_names = list(feature_names)

print("\nLoaded Linear Regression model.")
print("Number of features:", len(feature_names))

# --------------------------------------------------
# 2. PREPARE CHRONOLOGICAL TEST DATA
# --------------------------------------------------

train = df[df["YearMonth"] < TEST_START].copy()
test = df[df["YearMonth"] >= TEST_START].copy()

X_test = test[feature_names].astype(float)
y_test = test[TARGET].astype(float)

print("\nTraining records:", len(train))
print("Testing records:", len(test))

print(
    "Test period:",
    test["YearMonth"].min().date(),
    "to",
    test["YearMonth"].max().date()
)

# --------------------------------------------------
# 3. VERIFY EXISTING MODEL
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nFINAL LINEAR REGRESSION TEST PERFORMANCE")
print("-" * 50)
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))

# --------------------------------------------------
# 4. VERIFY AGAINST ORIGINAL EVALUATION REPORT
# --------------------------------------------------

if PREDICTION_PATH.exists():

    original = pd.read_csv(PREDICTION_PATH)
    original["YearMonth"] = pd.to_datetime(original["YearMonth"])

    original = original.sort_values("YearMonth").reset_index(drop=True)

    original_linear = original["Linear_Regression"].to_numpy()

    if len(original_linear) == len(predictions):

        max_difference = np.max(
            np.abs(original_linear - predictions)
        )

        print("\nOriginal model prediction verification:")
        print("Maximum prediction difference:", max_difference)

        if np.allclose(
            original_linear,
            predictions,
            rtol=1e-5,
            atol=1e-5
        ):
            print("Verification: PASSED")
        else:
            raise ValueError(
                "Predictions do not match the original evaluation report."
            )

    else:
        raise ValueError(
            "Prediction report and test dataset have different lengths."
        )

# --------------------------------------------------
# 5. SAVE FINAL DEPLOYMENT MODEL
# --------------------------------------------------

final_model_path = (
    FINAL_MODEL_DIR / "operational_pressure_final_model.pkl"
)

final_feature_path = (
    FINAL_MODEL_DIR / "operational_pressure_final_features.pkl"
)

joblib.dump(model, final_model_path)
joblib.dump(feature_names, final_feature_path)

metadata = {
    "project": "AeroCast AI",
    "module": "Operational Pressure Forecasting",
    "selected_model": "Linear Regression",
    "selection_method": "Four-fold chronological walk-forward validation",
    "target": TARGET,
    "test_start": TEST_START,
    "training_records": int(len(train)),
    "testing_records": int(len(test)),
    "features": feature_names,
    "test_metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    },
    "model_path": str(final_model_path),
    "feature_path": str(final_feature_path)
}

with open(
    FINAL_MODEL_DIR / "operational_pressure_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)

# --------------------------------------------------
# 6. SAVE FINAL PREDICTION REPORT
# --------------------------------------------------

final_predictions = pd.DataFrame({
    "YearMonth": test["YearMonth"].to_numpy(),
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "Residual": y_test.to_numpy() - predictions
})

final_predictions.to_csv(
    REPORT_DIR / "operational_pressure_final_predictions.csv",
    index=False
)

# --------------------------------------------------
# 7. FINAL VERIFICATION
# --------------------------------------------------

loaded_model = joblib.load(final_model_path)
loaded_features = joblib.load(final_feature_path)

reloaded_predictions = loaded_model.predict(
    test[loaded_features].astype(float)
)

if np.allclose(
    predictions,
    reloaded_predictions,
    rtol=1e-5,
    atol=1e-5
):
    print("\nSaved model verification: PASSED")
else:
    raise ValueError("Saved model verification failed.")

print("\n" + "=" * 65)
print("FINAL MODEL PREPARATION COMPLETED")
print("=" * 65)

print("Final model:", final_model_path)
print("Final features:", final_feature_path)
print("Metadata: operational_pressure_metadata.json")
print("Prediction report: operational_pressure_final_predictions.csv")
