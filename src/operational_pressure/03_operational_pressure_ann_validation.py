
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/operational_pressure_features.csv"
REPORT_DIR = ROOT / "reports/operational_pressure"
MODEL_DIR = ROOT / "models/operational_pressure"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Operational Pressure Index"

FEATURES = [
    "Year", "Month", "Quarter",
    "Month_Sin", "Month_Cos", "Time_Index",
    "Pressure_Lag_1", "Pressure_Lag_2",
    "Pressure_Lag_3", "Pressure_Lag_6",
    "Pressure_Lag_12",
    "Pressure_Rolling_Mean_3",
    "Pressure_Rolling_Mean_6",
    "Pressure_Rolling_Mean_12",
    "Pressure_Rolling_Std_3",
    "Pressure_Rolling_Std_6",
    "Pressure_Rolling_Std_12",
    "Passenger_Lag_1", "Passenger_Lag_12",
    "Aircraft_Lag_1", "Aircraft_Lag_12",
    "Cargo_Lag_1", "Cargo_Lag_12"
]

df = pd.read_csv(DATA_PATH)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

# Use training period only. The final 63 months remain untouched.
train = df[df["YearMonth"] < "2021-06-01"].reset_index(drop=True)

X = train[FEATURES].astype(float)
y = train[TARGET].astype(float)

tscv = TimeSeriesSplit(n_splits=4, test_size=24)

results = []

print("=" * 65)
print("OPERATIONAL PRESSURE ANN WALK-FORWARD VALIDATION")
print("=" * 65)

for fold, (fit_idx, val_idx) in enumerate(tscv.split(X), start=1):

    print(f"\nFOLD {fold}")
    print("-" * 50)

    X_fold_train = X.iloc[fit_idx]
    X_fold_val = X.iloc[val_idx]

    y_fold_train = y.iloc[fit_idx]
    y_fold_val = y.iloc[val_idx]

    # Keep a chronological validation segment for early stopping
    split_index = int(len(X_fold_train) * 0.80)

    X_fit = X_fold_train.iloc[:split_index]
    X_early_val = X_fold_train.iloc[split_index:]

    y_fit = y_fold_train.iloc[:split_index]
    y_early_val = y_fold_train.iloc[split_index:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_fit_scaled = x_scaler.fit_transform(X_fit)
    X_early_val_scaled = x_scaler.transform(X_early_val)
    X_fold_val_scaled = x_scaler.transform(X_fold_val)

    y_fit_scaled = y_scaler.fit_transform(
        y_fit.to_numpy().reshape(-1, 1)
    )

    y_early_val_scaled = y_scaler.transform(
        y_early_val.to_numpy().reshape(-1, 1)
    )

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED + fold)

    model = Sequential([
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.001),
            input_shape=(len(FEATURES),)
        ),
        Dropout(0.15),

        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),
        Dropout(0.10),

        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

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

    model.fit(
        X_fit_scaled,
        y_fit_scaled,
        validation_data=(
            X_early_val_scaled,
            y_early_val_scaled
        ),
        epochs=300,
        batch_size=16,
        shuffle=False,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    predictions_scaled = model.predict(
        X_fold_val_scaled,
        verbose=0
    )

    predictions = y_scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    mae = mean_absolute_error(y_fold_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_fold_val, predictions)
    )

    r2 = r2_score(y_fold_val, predictions)

    results.append({
        "Fold": fold,
        "Training_Records": len(fit_idx),
        "Validation_Records": len(val_idx),
        "Validation_Start": train.loc[val_idx[0], "YearMonth"],
        "Validation_End": train.loc[val_idx[-1], "YearMonth"],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("Training records:", len(fit_idx))
    print("Validation records:", len(val_idx))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "operational_pressure_ann_walk_forward.csv",
    index=False
)

summary = pd.DataFrame([{
    "Average_MAE": results_df["MAE"].mean(),
    "Average_RMSE": results_df["RMSE"].mean(),
    "Average_R2": results_df["R2"].mean(),
    "Std_MAE": results_df["MAE"].std()
}])

summary.to_csv(
    REPORT_DIR / "operational_pressure_ann_validation_summary.csv",
    index=False
)

print("\n" + "=" * 65)
print("ANN WALK-FORWARD VALIDATION RESULTS")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nAVERAGE VALIDATION PERFORMANCE")
print(summary.round(4).to_string(index=False))

print("\nValidation report saved in:", REPORT_DIR)
