import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_path = Path(__file__).resolve().parents[2]

input_path = project_path / "data" / "features" / "passenger_features.csv"

model_folder = project_path / "models" / "random_forest"
report_folder = project_path / "reports" / "evaluation"

model_folder.mkdir(parents=True, exist_ok=True)
report_folder.mkdir(parents=True, exist_ok=True)

print("\n========== AEROCAST AI ==========")
print("RANDOM FOREST REGRESSION")
print("================================")

if not input_path.exists():
    raise FileNotFoundError("Run Task 14 first.")

df = pd.read_csv(input_path)

target = "Passenger Count"

X = df.drop(columns=[target, "YearMonth"], errors="ignore")
y = df[target]

X = X.apply(pd.to_numeric, errors="coerce")

valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]

split_index = int(len(df.loc[valid_rows]) * 0.80)

if split_index < 2 or len(X) - split_index < 2:
    raise ValueError("Not enough records for training and testing.")

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nDataset Information")
print("Total Records:", len(X))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
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

# Save model
joblib.dump(
    model,
    model_folder / "random_forest_model.pkl"
)

# Save predictions
results = pd.DataFrame({
    "Actual Passenger Count": y_test.values,
    "Predicted Passenger Count": predictions
})

results.to_csv(
    report_folder / "random_forest_predictions.csv",
    index=False
)

# Save evaluation metrics
metrics = pd.DataFrame({
    "Model": ["Random Forest"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2 Score": [r2]
})

metrics.to_csv(
    report_folder / "random_forest_metrics.csv",
    index=False
)

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance.to_csv(
    report_folder / "random_forest_feature_importance.csv",
    index=False
)

print("\nTop 10 Important Features")
print(importance.head(10).to_string(index=False))

print("\nFirst 10 Predictions")
print(results.head(10).to_string(index=False))

print("\nSaved Files:")
print("random_forest_model.pkl")
print("random_forest_predictions.csv")
print("random_forest_metrics.csv")
print("random_forest_feature_importance.csv")

print("\n========== RANDOM FOREST COMPLETED ==========")
