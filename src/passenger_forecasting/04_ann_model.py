
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# Project paths
project_path = Path(__file__).resolve().parents[2]

input_path = project_path / "data" / "features" / "passenger_features.csv"

model_folder = project_path / "models" / "ann"
report_folder = project_path / "reports" / "evaluation"

model_folder.mkdir(parents=True, exist_ok=True)
report_folder.mkdir(parents=True, exist_ok=True)


print("\n========== AEROCAST AI ==========")
print("ARTIFICIAL NEURAL NETWORK")
print("================================")


# Load dataset
if not input_path.exists():
    raise FileNotFoundError("Run Task 14 first.")

df = pd.read_csv(input_path)

# Maintain chronological order
df = df.sort_values("YearMonth").reset_index(drop=True)

target = "Passenger Count"

X = df.drop(columns=[target, "YearMonth"], errors="ignore")
y = df[target]

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

valid_rows = X.notna().all(axis=1) & y.notna()

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)


# Chronological train-test split
split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

if len(X_train) < 10 or len(X_test) < 2:
    raise ValueError("Not enough records for ANN training.")


# Separate training and validation data
validation_index = int(len(X_train) * 0.80)

X_fit = X_train.iloc[:validation_index]
X_val = X_train.iloc[validation_index:]

y_fit = y_train.iloc[:validation_index]
y_val = y_train.iloc[validation_index:]


print("\nDataset Information")
print("Total Records:", len(X))
print("Training Records:", len(X_train))
print("ANN Training Records:", len(X_fit))
print("Validation Records:", len(X_val))
print("Testing Records:", len(X_test))


# Scale input features
X_scaler = StandardScaler()

X_fit_scaled = X_scaler.fit_transform(X_fit)
X_val_scaled = X_scaler.transform(X_val)
X_test_scaled = X_scaler.transform(X_test)


# Scale target values
y_scaler = StandardScaler()

y_fit_scaled = y_scaler.fit_transform(
    y_fit.to_numpy().reshape(-1, 1)
)

y_val_scaled = y_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
)


# Build ANN model
model = Sequential([
    Input(shape=(X_fit_scaled.shape[1],)),

    Dense(64, activation="relu"),
    Dropout(0.1),

    Dense(32, activation="relu"),

    Dense(1)
])


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mae"]
)


# Stop when validation performance stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=8,
    min_lr=0.00001
)


# Train ANN
history = model.fit(
    X_fit_scaled,
    y_fit_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=200,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=False
)


# Generate predictions
predictions_scaled = model.predict(X_test_scaled)

predictions = y_scaler.inverse_transform(
    predictions_scaled
).flatten()

predictions = np.maximum(predictions, 0)


# Evaluate model
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


print("\n========== MODEL PERFORMANCE ==========")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# Save ANN model
model.save(
    model_folder / "ann_passenger_model.keras"
)


# Save both scalers
joblib.dump(
    X_scaler,
    model_folder / "ann_scaler.pkl"
)

joblib.dump(
    y_scaler,
    model_folder / "ann_target_scaler.pkl"
)


# Save predictions
results = pd.DataFrame({
    "Actual Passenger Count": y_test.values,
    "Predicted Passenger Count": predictions
})

results.to_csv(
    report_folder / "ann_predictions.csv",
    index=False
)


# Save evaluation metrics
metrics = pd.DataFrame({
    "Model": ["Artificial Neural Network"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2 Score": [r2]
})

metrics.to_csv(
    report_folder / "ann_metrics.csv",
    index=False
)


# Save training history
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    report_folder / "ann_training_history.csv",
    index=False
)


# Display predictions
print("\nFirst 10 Predictions")

print(
    results.head(10).to_string(index=False)
)


print("\nSaved Files:")
print("ann_passenger_model.keras")
print("ann_scaler.pkl")
print("ann_target_scaler.pkl")
print("ann_predictions.csv")
print("ann_metrics.csv")
print("ann_training_history.csv")

print("\n========== ANN TRAINING COMPLETED ==========")
