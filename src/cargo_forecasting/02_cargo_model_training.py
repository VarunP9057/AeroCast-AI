
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/features/cargo_monthly_features.csv"

MODEL_DIR = ROOT / "models/cargo_forecasting"
REPORT_DIR = ROOT / "reports/cargo_forecasting"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

print("\nDataset Shape:", df.shape)


# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

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

TARGET = "Cargo Weight LBS"

X = df[FEATURES].values
y = df[TARGET].values

dates = df["YearMonth"]


# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

train_dates = dates[:split_index]
test_dates = dates[split_index:]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nTraining Period:")
print(train_dates.iloc[0], "to", train_dates.iloc[-1])

print("\nTesting Period:")
print(test_dates.iloc[0], "to", test_dates.iloc[-1])


# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
)

# Scale target for ANN training.
# Linear Regression and Random Forest use original target values.


# --------------------------------------------------
# 6. LINEAR REGRESSION
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_train)

linear_predictions = linear_model.predict(X_test_scaled)


# --------------------------------------------------
# 7. RANDOM FOREST
# --------------------------------------------------

random_forest_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

random_forest_predictions = random_forest_model.predict(X_test)


# --------------------------------------------------
# 8. ANN MODEL
# --------------------------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

ann_model = Sequential([
    Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(0.001),
        input_shape=(X_train_scaled.shape[1],)
    ),
    Dropout(0.15),

    Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.10),

    Dense(
        16,
        activation="relu"
    ),

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
    X_train_scaled,
    y_train_scaled,
    validation_split=0.20,
    epochs=250,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    shuffle=False,
    verbose=1
)

ann_predictions_scaled = ann_model.predict(
    X_test_scaled,
    verbose=0
)

ann_predictions = y_scaler.inverse_transform(
    ann_predictions_scaled
).flatten()


# --------------------------------------------------
# 9. EVALUATION FUNCTION
# --------------------------------------------------

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
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape
    }


# --------------------------------------------------
# 10. MODEL COMPARISON
# --------------------------------------------------

predictions = {
    "Linear Regression": linear_predictions,
    "Random Forest": random_forest_predictions,
    "ANN": ann_predictions
}

results = []

for model_name, model_predictions in predictions.items():

    metrics = calculate_metrics(
        y_test,
        model_predictions
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


# --------------------------------------------------
# 11. SAVE PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "YearMonth": test_dates.values,
    "Actual": y_test,
    "Linear_Regression": linear_predictions,
    "Random_Forest": random_forest_predictions,
    "ANN": ann_predictions
})

prediction_df.to_csv(
    REPORT_DIR / "cargo_model_predictions.csv",
    index=False
)


# --------------------------------------------------
# 12. SAVE MODEL COMPARISON
# --------------------------------------------------

comparison.to_csv(
    REPORT_DIR / "cargo_model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 13. SAVE TRAINING HISTORY
# --------------------------------------------------

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    REPORT_DIR / "cargo_ann_training_history.csv",
    index=False
)


# --------------------------------------------------
# 14. SAVE MODELS AND SCALERS
# --------------------------------------------------

with open(MODEL_DIR / "cargo_linear_regression.pkl", "wb") as f:
    pickle.dump(linear_model, f)

with open(MODEL_DIR / "cargo_random_forest.pkl", "wb") as f:
    pickle.dump(random_forest_model, f)

ann_model.save(
    MODEL_DIR / "cargo_ann_model.keras"
)

with open(MODEL_DIR / "cargo_ann_scaler.pkl", "wb") as f:
    pickle.dump(X_scaler, f)

with open(MODEL_DIR / "cargo_ann_target_scaler.pkl", "wb") as f:
    pickle.dump(y_scaler, f)

with open(MODEL_DIR / "cargo_feature_names.pkl", "wb") as f:
    pickle.dump(FEATURES, f)


# --------------------------------------------------
# 15. SAVE TRAINING SUMMARY
# --------------------------------------------------

best_model = comparison.iloc[0]

summary = f"""
AEROCAST AI
CARGO DEMAND FORECASTING - EXPERIMENT 1

Dataset Records: {len(df)}
Features: {len(FEATURES)}

Training Records: {len(X_train)}
Testing Records: {len(X_test)}

Training Period:
{train_dates.iloc[0]} to {train_dates.iloc[-1]}

Testing Period:
{test_dates.iloc[0]} to {test_dates.iloc[-1]}

Best Model by MAE:
{best_model['Model']}

MAE: {best_model['MAE']:.2f}
RMSE: {best_model['RMSE']:.2f}
R2: {best_model['R2']:.4f}
MAPE: {best_model['MAPE']:.2f}%
WAPE: {best_model['WAPE']:.2f}%

Models trained:
Linear Regression
Random Forest
Artificial Neural Network
"""

(REPORT_DIR / "cargo_model_training_summary.txt").write_text(
    summary,
    encoding="utf-8"
)


# --------------------------------------------------
# 16. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 65)
print("CARGO FORECASTING EXPERIMENT 1 COMPLETED")
print("=" * 65)

print("\nMODEL COMPARISON:")
print(comparison.to_string(index=False))

print("\nBest Model:", best_model["Model"])
print("MAE:", round(best_model["MAE"], 2))
print("RMSE:", round(best_model["RMSE"], 2))
print("R2:", round(best_model["R2"], 4))
print("MAPE:", round(best_model["MAPE"], 2), "%")
print("WAPE:", round(best_model["WAPE"], 2), "%")

print("\nModels Saved:", MODEL_DIR)
print("Reports Saved:", REPORT_DIR)

print("\nExperiment 1 completed successfully.")
