
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_path = Path(__file__).resolve().parents[2]

input_path = project_path / "data" / "features" / "passenger_features.csv"

model_folder = project_path / "models" / "linear_regression"
report_folder = project_path / "reports" / "evaluation"

model_folder.mkdir(parents=True, exist_ok=True)
report_folder.mkdir(parents=True, exist_ok=True)

print("\n========== AEROCAST AI ==========")
print("LINEAR REGRESSION MODEL")
print("================================")

if not input_path.exists():
    raise FileNotFoundError(
        "Feature dataset not found. Run Task 14 first."
    )

df = pd.read_csv(input_path)

target = "Passenger Count"

if target not in df.columns:
    raise ValueError("Passenger Count column not found.")

# Select input features
excluded_columns = [target, "YearMonth"]

X = df.drop(columns=excluded_columns, errors="ignore")
y = df[target]

# Ensure all input features are numeric
X = X.apply(pd.to_numeric, errors="coerce")

valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]
df = df.loc[valid_rows].reset_index(drop=True)

# Chronological 80-20 split
split_index = int(len(df) * 0.80)

if split_index < 2 or len(df) - split_index < 2:
    raise ValueError("Not enough records for training and testing.")

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nDataset Information")
print("Total Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict passenger counts
predictions = model.predict(X_test)

predictions = np.maximum(predictions, 0)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save trained model
joblib.dump(
    model,
    model_folder / "linear_regression_model.pkl"
)

# Save predictions
results = pd.DataFrame({
    "Actual Passenger Count": y_test.values,
    "Predicted Passenger Count": predictions
})

results.to_csv(
    report_folder / "linear_regression_predictions.csv",
    index=False
)

# Save evaluation metrics
metrics = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2 Score": [r2]
})

metrics.to_csv(
    report_folder / "linear_regression_metrics.csv",
    index=False
)

print("\nFirst 10 Predictions")
print(results.head(10).to_string(index=False))

print("\nSaved Files:")
print("linear_regression_model.pkl")
print("linear_regression_predictions.csv")
print("linear_regression_metrics.csv")

print("\n========== LINEAR REGRESSION COMPLETED ==========")
