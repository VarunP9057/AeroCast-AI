
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/cargo_advanced_features.csv"
MODEL_DIR = ROOT / "models/cargo_forecasting_experiment3"
REPORT_DIR = ROOT / "reports/cargo_forecasting_experiment3"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Cargo Weight LBS"

FEATURES = [
    "Year", "Month", "Quarter",
    "Month_Sin", "Month_Cos", "Time_Index",
    "Lag_1", "Lag_2", "Lag_3",
    "Lag_6", "Lag_12", "Lag_18", "Lag_24",
    "Rolling_Mean_3", "Rolling_Mean_6",
    "Rolling_Mean_12", "Rolling_Mean_24",
    "Rolling_Std_3", "Rolling_Std_6", "Rolling_Std_12",
    "Previous_Year_Change", "Previous_Year_Change_Pct",
    "Trend_Slope_6", "Trend_Slope_12"
]

df = pd.read_csv(DATA_PATH)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

train = df[df["YearMonth"] < "2021-06-01"].copy()
test = df[df["YearMonth"] >= "2021-06-01"].copy()

X_train = train[FEATURES].astype(float).values
y_train = train[TARGET].astype(float).values

X_test = test[FEATURES].astype(float).values
y_test = test[TARGET].astype(float).values

print("CARGO EXPERIMENT 3")
print("=" * 55)
print("Training records:", len(train))
print("Testing records:", len(test))
print("Features:", len(FEATURES))
print("Training period:", train["YearMonth"].min(), "to", train["YearMonth"].max())
print("Testing period:", test["YearMonth"].min(), "to", test["YearMonth"].max())

if np.isnan(X_train).any() or np.isnan(X_test).any():
    raise ValueError("Missing values found in feature data.")

tscv = TimeSeriesSplit(n_splits=4, test_size=24)

ARCHITECTURES = {
    "Compact_ANN": {
        "layers": [32, 16],
        "dropout": 0.10,
        "l2": 0.001,
        "learning_rate": 0.001
    },
    "Balanced_ANN": {
        "layers": [64, 32, 16],
        "dropout": 0.15,
        "l2": 0.001,
        "learning_rate": 0.0005
    },
    "Regularized_ANN": {
        "layers": [128, 64, 32],
        "dropout": 0.25,
        "l2": 0.005,
        "learning_rate": 0.0005
    }
}


def build_model(input_size, config):
    model = Sequential()
    model.add(Input(shape=(input_size,)))

    for neurons in config["layers"]:
        model.add(
            Dense(
                neurons,
                activation="relu",
                kernel_regularizer=l2(config["l2"])
            )
        )
        model.add(Dropout(config["dropout"]))

    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"]
        ),
        loss="mse"
    )

    return model


validation_results = []

for model_name, config in ARCHITECTURES.items():

    print(f"\nTraining architecture: {model_name}")
    print("-" * 55)

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for fold, (fit_idx, val_idx) in enumerate(tscv.split(X_train), start=1):

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42 + fold)

        X_fit = X_train[fit_idx]
        X_val = X_train[val_idx]

        y_fit = y_train[fit_idx]
        y_val = y_train[val_idx]

        scaler = StandardScaler()
        X_fit_scaled = scaler.fit_transform(X_fit)
        X_val_scaled = scaler.transform(X_val)

        target_scaler = StandardScaler()
        y_fit_scaled = target_scaler.fit_transform(
            y_fit.reshape(-1, 1)
        )

        y_val_scaled = target_scaler.transform(
            y_val.reshape(-1, 1)
        )

        model = build_model(len(FEATURES), config)

        early_stop = EarlyStopping(
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
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=300,
            batch_size=16,
            shuffle=False,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        pred_scaled = model.predict(X_val_scaled, verbose=0)

        predictions = target_scaler.inverse_transform(
            pred_scaled
        ).flatten()

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        print(
            f"Fold {fold}: "
            f"MAE={mae:,.0f}, "
            f"RMSE={rmse:,.0f}, "
            f"R2={r2:.4f}"
        )

    validation_results.append({
        "Model": model_name,
        "Average_MAE": np.mean(fold_mae),
        "Average_RMSE": np.mean(fold_rmse),
        "Average_R2": np.mean(fold_r2)
    })

    print(f"{model_name} average MAE: {np.mean(fold_mae):,.0f}")


validation_df = pd.DataFrame(validation_results)
validation_df = validation_df.sort_values("Average_MAE")

validation_df.to_csv(
    REPORT_DIR / "cargo_experiment3_validation_results.csv",
    index=False
)

print("\nWALK-FORWARD ARCHITECTURE COMPARISON")
print("=" * 65)
print(validation_df.to_string(index=False))

best_architecture = validation_df.iloc[0]["Model"]

print("\nSelected architecture:", best_architecture)
print("Selection criterion: Lowest average walk-forward MAE")

print("\nExperiment 3 validation completed.")
print("The final 63-month test set has NOT been used.")
