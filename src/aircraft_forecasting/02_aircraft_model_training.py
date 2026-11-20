
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "features" / "aircraft_monthly_features.csv"
MODEL_FOLDER = PROJECT_ROOT / "models" / "aircraft_forecasting"
REPORT_FOLDER = PROJECT_ROOT / "reports" / "aircraft_forecasting"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

df = pd.read_csv(DATA_FILE)

TARGET = "Landing Count"

FEATURES = [
    "Year",
    "Month",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_6",
    "Lag_12",
    "Rolling_Mean_3",
    "Rolling_Mean_6",
    "Rolling_Std_3",
    "Rolling_Std_6"
]

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
dates = df["YearMonth"]

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = dates.iloc[split_index:]

validation_index = int(len(X_train) * 0.80)

X_fit = X_train.iloc[:validation_index]
X_validation = X_train.iloc[validation_index:]

y_fit = y_train.iloc[:validation_index]
y_validation = y_train.iloc[validation_index:]

print("\n========== AEROCAST AI ==========")
print("AIRCRAFT MOVEMENT MODEL TRAINING")
print("================================")

print("\nTotal records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("ANN fitting records:", len(X_fit))
print("ANN validation records:", len(X_validation))

print("\nTraining period:", dates.iloc[0], "to", dates.iloc[split_index - 1])
print("Testing period:", test_dates.iloc[0], "to", test_dates.iloc[-1])

# 1. Seasonal Naive Baseline

baseline_predictions = df["Lag_12"].iloc[split_index:].to_numpy()

# 2. Linear Regression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = np.maximum(
    linear_model.predict(X_test), 0
)

# 3. Random Forest

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = np.maximum(
    rf_model.predict(X_test), 0
)

# 4. Artificial Neural Network

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_fit_scaled = x_scaler.fit_transform(X_fit)
X_validation_scaled = x_scaler.transform(X_validation)
X_test_scaled = x_scaler.transform(X_test)

y_fit_scaled = y_scaler.fit_transform(
    y_fit.to_numpy().reshape(-1, 1)
)

y_validation_scaled = y_scaler.transform(
    y_validation.to_numpy().reshape(-1, 1)
)

ann_model = Sequential([
    tf.keras.Input(shape=(len(FEATURES),)),
    Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.15),
    Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.10),
    Dense(16, activation="relu"),
    Dense(1)
])

ann_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

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

history = ann_model.fit(
    X_fit_scaled,
    y_fit_scaled,
    validation_data=(X_validation_scaled, y_validation_scaled),
    epochs=250,
    batch_size=16,
    shuffle=False,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

ann_scaled_predictions = ann_model.predict(
    X_test_scaled,
    verbose=0
)

ann_predictions = y_scaler.inverse_transform(
    ann_scaled_predictions
).flatten()

ann_predictions = np.maximum(ann_predictions, 0)

# Evaluation

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    nonzero = np.asarray(actual) != 0
    mape = np.mean(
        np.abs(
            (np.asarray(actual)[nonzero] -
             np.asarray(predicted)[nonzero]) /
            np.asarray(actual)[nonzero]
        )
    ) * 100

    wape = (
        np.sum(np.abs(np.asarray(actual) - np.asarray(predicted))) /
        np.sum(np.abs(actual))
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape
    }


predictions = {
    "Seasonal Naive": baseline_predictions,
    "Linear Regression": linear_predictions,
    "Random Forest": rf_predictions,
    "ANN": ann_predictions
}

results = []

for model_name, model_predictions in predictions.items():
    metrics = calculate_metrics(y_test.to_numpy(), model_predictions)

    results.append({
        "Model": model_name,
        **metrics
    })

metrics_df = pd.DataFrame(results)
metrics_df = metrics_df.sort_values("MAE").reset_index(drop=True)
metrics_df.insert(0, "Rank", range(1, len(metrics_df) + 1))

print("\n========== MODEL COMPARISON ==========")
print(metrics_df.to_string(index=False))

# Save predictions

prediction_data = pd.DataFrame({
    "YearMonth": test_dates.to_numpy(),
    "Actual": y_test.to_numpy(),
    "Seasonal_Naive": baseline_predictions,
    "Linear_Regression": linear_predictions,
    "Random_Forest": rf_predictions,
    "ANN": ann_predictions
})

prediction_data.to_csv(
    REPORT_FOLDER / "aircraft_model_predictions.csv",
    index=False
)

metrics_df.to_csv(
    REPORT_FOLDER / "aircraft_model_comparison.csv",
    index=False
)

pd.DataFrame(history.history).to_csv(
    REPORT_FOLDER / "aircraft_ann_training_history.csv",
    index=False
)

# Save trained models and scalers

joblib.dump(
    linear_model,
    MODEL_FOLDER / "aircraft_linear_regression.pkl"
)

joblib.dump(
    rf_model,
    MODEL_FOLDER / "aircraft_random_forest.pkl"
)

ann_model.save(
    MODEL_FOLDER / "aircraft_ann_model.keras"
)

joblib.dump(
    x_scaler,
    MODEL_FOLDER / "aircraft_ann_scaler.pkl"
)

joblib.dump(
    y_scaler,
    MODEL_FOLDER / "aircraft_ann_target_scaler.pkl"
)

joblib.dump(
    FEATURES,
    MODEL_FOLDER / "aircraft_feature_names.pkl"
)

# Save final summary

best_model = metrics_df.iloc[0]

summary = [
    "AEROCAST AI - AIRCRAFT MOVEMENT FORECASTING",
    "============================================",
    f"Total records: {len(df)}",
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    f"Training period: {dates.iloc[0]} to {dates.iloc[split_index - 1]}",
    f"Testing period: {test_dates.iloc[0]} to {test_dates.iloc[-1]}",
    "",
    "MODEL COMPARISON",
    metrics_df.to_string(index=False),
    "",
    f"Best model based on MAE: {best_model['Model']}",
    f"Best MAE: {best_model['MAE']:.2f}"
]

with open(
    REPORT_FOLDER / "aircraft_model_training_summary.txt",
    "w"
) as file:
    file.write("\n".join(summary))

print("\nModels saved in:", MODEL_FOLDER)
print("Reports saved in:", REPORT_FOLDER)

print("\n========== AIRCRAFT MODEL TRAINING COMPLETED ==========")
