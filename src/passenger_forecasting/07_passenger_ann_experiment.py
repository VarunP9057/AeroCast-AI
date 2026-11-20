
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# =====================================================
# 1. CONFIGURATION
# =====================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/features/passenger_features.csv"

MODEL_DIR = ROOT / "models/ann_experiment"
REPORT_DIR = ROOT / "reports/ann_experiment"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Passenger Count"

print("\n==========================================")
print("AEROCAST AI - ANN EXPERIMENT 2")
print("Passenger Volume Forecasting")
print("==========================================")


# =====================================================
# 2. LOAD AND VALIDATE DATA
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)

df = df.sort_values("YearMonth").reset_index(drop=True)

X = df.drop(columns=[TARGET, "YearMonth"])
y = pd.to_numeric(df[TARGET], errors="coerce")

X = X.apply(pd.to_numeric, errors="coerce")

valid = X.notna().all(axis=1) & y.notna()

df = df.loc[valid].reset_index(drop=True)
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

dates = df["YearMonth"].reset_index(drop=True)

if (y <= 0).any():
    raise ValueError("MAPE requires positive passenger counts.")

split = int(len(df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

test_dates = dates.iloc[split:]

print("\nDATA INFORMATION")
print("Total records:", len(df))
print("Features:", X.shape[1])
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Test start:", test_dates.iloc[0])
print("Test end:", test_dates.iloc[-1])


# =====================================================
# 3. TRAINING AND VALIDATION SPLIT
# =====================================================

validation_index = int(len(X_train) * 0.80)

X_fit = X_train.iloc[:validation_index]
X_val = X_train.iloc[validation_index:]

y_fit = y_train.iloc[:validation_index]
y_val = y_train.iloc[validation_index:]

print("\nTRAINING DETAILS")
print("ANN fitting records:", len(X_fit))
print("Validation records:", len(X_val))


# =====================================================
# 4. FEATURE AND TARGET SCALING
# =====================================================

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_fit_scaled = X_scaler.fit_transform(X_fit)
X_val_scaled = X_scaler.transform(X_val)
X_test_scaled = X_scaler.transform(X_test)

y_fit_scaled = y_scaler.fit_transform(
    y_fit.to_numpy().reshape(-1, 1)
)

y_val_scaled = y_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
)


# =====================================================
# 5. BUILD IMPROVED ANN
# =====================================================

model = Sequential([
    Input(shape=(X_fit_scaled.shape[1],)),

    Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    Dropout(0.20),

    Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    Dropout(0.10),

    Dense(32, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()


# =====================================================
# 6. CALLBACKS
# =====================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=0.00001
)


# =====================================================
# 7. TRAIN ANN
# =====================================================

history = model.fit(
    X_fit_scaled,
    y_fit_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=300,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    shuffle=False,
    verbose=1
)


# =====================================================
# 8. PREDICTIONS
# =====================================================

pred_scaled = model.predict(
    X_test_scaled,
    verbose=0
)

predictions = y_scaler.inverse_transform(
    pred_scaled
).flatten()

predictions = np.maximum(predictions, 0)


# =====================================================
# 9. SEASONAL NAIVE BASELINE
# =====================================================

seasonal_predictions = X_test["Lag_12"].to_numpy()

if not np.isfinite(seasonal_predictions).all():
    raise ValueError("Invalid values in seasonal baseline.")


# =====================================================
# 10. EVALUATION FUNCTION
# =====================================================

def evaluate(actual, predicted):

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


ann_metrics = evaluate(y_test.to_numpy(), predictions)

seasonal_metrics = evaluate(
    y_test.to_numpy(),
    seasonal_predictions
)


# =====================================================
# 11. SAVE PREDICTIONS
# =====================================================

prediction_results = pd.DataFrame({
    "YearMonth": test_dates.to_numpy(),
    "Actual Passenger Count": y_test.to_numpy(),
    "ANN Predicted Passenger Count": predictions,
    "Seasonal Naive Prediction": seasonal_predictions,
    "ANN Absolute Error": np.abs(y_test.to_numpy() - predictions),
    "Seasonal Absolute Error": np.abs(
        y_test.to_numpy() - seasonal_predictions
    )
})

prediction_results.to_csv(
    REPORT_DIR / "ann_experiment_predictions.csv",
    index=False
)


# =====================================================
# 12. SAVE METRICS
# =====================================================

metrics_df = pd.DataFrame([
    {
        "Model": "Improved ANN",
        **ann_metrics
    },
    {
        "Model": "Seasonal Naive",
        **seasonal_metrics
    }
])

metrics_df.to_csv(
    REPORT_DIR / "ann_experiment_metrics.csv",
    index=False
)


# =====================================================
# 13. SAVE TRAINING HISTORY
# =====================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    REPORT_DIR / "ann_experiment_training_history.csv",
    index=False
)


# =====================================================
# 14. SAVE MODEL AND SCALERS
# =====================================================

model.save(
    MODEL_DIR / "ann_experiment_model.keras"
)

joblib.dump(
    X_scaler,
    MODEL_DIR / "ann_experiment_scaler.pkl"
)

joblib.dump(
    y_scaler,
    MODEL_DIR / "ann_experiment_target_scaler.pkl"
)

joblib.dump(
    list(X.columns),
    MODEL_DIR / "ann_experiment_features.pkl"
)


# =====================================================
# 15. TRAINING LOSS VISUALIZATION
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Experiment 2 - Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "ann_experiment_training_loss.png",
    dpi=300
)

plt.close()


# =====================================================
# 16. ACTUAL VS PREDICTED VISUALIZATION
# =====================================================

plt.figure(figsize=(14, 6))

plt.plot(
    test_dates,
    y_test.to_numpy(),
    label="Actual Passengers",
    linewidth=2
)

plt.plot(
    test_dates,
    predictions,
    label="Improved ANN",
    linewidth=2
)

plt.plot(
    test_dates,
    seasonal_predictions,
    label="Seasonal Naive",
    alpha=0.8
)

plt.title("Passenger Traffic - Actual vs Predicted")
plt.xlabel("Month")
plt.ylabel("Passenger Count")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "ann_experiment_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# =====================================================
# 17. ERROR DISTRIBUTION
# =====================================================

errors = y_test.to_numpy() - predictions

plt.figure(figsize=(9, 5))

plt.hist(errors, bins=20)

plt.title("ANN Experiment 2 - Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "ann_experiment_error_distribution.png",
    dpi=300
)

plt.close()


# =====================================================
# 18. FINAL RESULTS
# =====================================================

print("\n==========================================")
print("ANN EXPERIMENT 2 - RESULTS")
print("==========================================")

print(metrics_df.to_string(index=False))

print("\nFIRST 10 PREDICTIONS")

print(
    prediction_results.head(10).to_string(index=False)
)

print("\nBEST VALIDATION EPOCH:")

print(
    int(np.argmin(history.history["val_loss"])) + 1
)

print("\nTRAINED EPOCHS:", len(history.history["loss"]))

print("\nSAVED MODEL:")
print(MODEL_DIR / "ann_experiment_model.keras")

print("\nSAVED REPORTS:")
for file in sorted(REPORT_DIR.iterdir()):
    print(file.name)

print("\n==========================================")
print("ANN EXPERIMENT 2 COMPLETED")
print("==========================================")
