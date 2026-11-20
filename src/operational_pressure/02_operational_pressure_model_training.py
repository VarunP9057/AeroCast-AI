
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/operational_pressure_features.csv"
MODEL_DIR = ROOT / "models/operational_pressure"
REPORT_DIR = ROOT / "reports/operational_pressure"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Operational Pressure Index"

FEATURES = [
    "Year",
    "Month",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Time_Index",
    "Pressure_Lag_1",
    "Pressure_Lag_2",
    "Pressure_Lag_3",
    "Pressure_Lag_6",
    "Pressure_Lag_12",
    "Pressure_Rolling_Mean_3",
    "Pressure_Rolling_Mean_6",
    "Pressure_Rolling_Mean_12",
    "Pressure_Rolling_Std_3",
    "Pressure_Rolling_Std_6",
    "Pressure_Rolling_Std_12",
    "Passenger_Lag_1",
    "Passenger_Lag_12",
    "Aircraft_Lag_1",
    "Aircraft_Lag_12",
    "Cargo_Lag_1",
    "Cargo_Lag_12"
]

df = pd.read_csv(DATA_PATH)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)

train_mask = df["YearMonth"] < "2021-06-01"
test_mask = df["YearMonth"] >= "2021-06-01"

X_train = X.loc[train_mask].reset_index(drop=True)
y_train = y.loc[train_mask].reset_index(drop=True)

X_test = X.loc[test_mask].reset_index(drop=True)
y_test = y.loc[test_mask].reset_index(drop=True)

train_dates = df.loc[train_mask, "YearMonth"].reset_index(drop=True)
test_dates = df.loc[test_mask, "YearMonth"].reset_index(drop=True)

print("=" * 65)
print("OPERATIONAL PRESSURE MODEL TRAINING")
print("=" * 65)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", len(FEATURES))
print("Training period:", train_dates.min(), "to", train_dates.max())
print("Testing period:", test_dates.min(), "to", test_dates.max())

# Seasonal naive baseline
seasonal_predictions = X_test["Pressure_Lag_12"].to_numpy()

# Traditional model configurations
models = {
    "Linear_Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression()
    ),

    "Random_Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        )
    ),

    "Extra_Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        )
    ),

    "Gradient_Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
            random_state=SEED
        )
    )
}

# Walk-forward validation on training data only
tscv = TimeSeriesSplit(n_splits=4, test_size=24)

cv_results = []

print("\nStarting walk-forward validation...")

for model_name, model in models.items():

    fold_maes = []
    fold_rmses = []
    fold_r2s = []

    for fold, (fit_idx, val_idx) in enumerate(
        tscv.split(X_train), start=1
    ):

        X_fit = X_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]

        y_fit = y_train.iloc[fit_idx]
        y_val = y_train.iloc[val_idx]

        model.fit(X_fit, y_fit)
        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)

        print(
            f"{model_name} | Fold {fold} | "
            f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}"
        )

    cv_results.append({
        "Model": model_name,
        "Average_MAE": np.mean(fold_maes),
        "Average_RMSE": np.mean(fold_rmses),
        "Average_R2": np.mean(fold_r2s)
    })

# ANN with chronological fit and validation split
print("\nTraining ANN...")

split_index = int(len(X_train) * 0.80)

X_fit = X_train.iloc[:split_index]
X_val = X_train.iloc[split_index:]

y_fit = y_train.iloc[:split_index]
y_val = y_train.iloc[split_index:]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_fit_scaled = x_scaler.fit_transform(X_fit)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_fit_scaled = y_scaler.fit_transform(
    y_fit.to_numpy().reshape(-1, 1)
)
y_val_scaled = y_scaler.transform(
    y_val.to_numpy().reshape(-1, 1)
)

ann = Sequential([
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

ann.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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

history = ann.fit(
    X_fit_scaled,
    y_fit_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=300,
    batch_size=16,
    shuffle=False,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

ann.save(MODEL_DIR / "operational_pressure_ann.keras")

with open(MODEL_DIR / "operational_pressure_x_scaler.pkl", "wb") as f:
    pickle.dump(x_scaler, f)

with open(MODEL_DIR / "operational_pressure_y_scaler.pkl", "wb") as f:
    pickle.dump(y_scaler, f)

with open(MODEL_DIR / "operational_pressure_feature_names.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

pd.DataFrame(history.history).to_csv(
    REPORT_DIR / "operational_pressure_ann_training_history.csv",
    index=False
)

# Save CV results
cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(
    REPORT_DIR / "operational_pressure_cross_validation.csv",
    index=False
)

print("\nWALK-FORWARD VALIDATION RESULTS")
print(cv_df.sort_values("Average_MAE").to_string(index=False))

# Fit traditional candidates on all training data
# Select candidates using validation MAE, not test MAE
fitted_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[model_name] = model

# Save fitted traditional models
for model_name, model in fitted_models.items():
    with open(MODEL_DIR / f"{model_name.lower()}.pkl", "wb") as f:
        pickle.dump(model, f)

# Evaluate candidates on the held-out test period
predictions_dict = {}

for model_name, model in fitted_models.items():
    predictions_dict[model_name] = model.predict(X_test)

predictions_dict["ANN"] = y_scaler.inverse_transform(
    ann.predict(X_test_scaled, verbose=0)
).flatten()

predictions_dict["Seasonal_Naive"] = seasonal_predictions

# Test metrics are reported for comparison only.
# Do not use test results to tune or select additional models.
comparison = []
prediction_data = {
    "YearMonth": test_dates,
    "Actual": y_test
}

for model_name, predictions in predictions_dict.items():

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    nonzero = np.abs(y_test.to_numpy()) > 1e-8

    mape = np.mean(
        np.abs(
            (y_test.to_numpy()[nonzero] - predictions[nonzero])
            / y_test.to_numpy()[nonzero]
        )
    ) * 100

    wape = (
        np.sum(np.abs(y_test.to_numpy() - predictions))
        / np.sum(np.abs(y_test.to_numpy()))
    ) * 100

    comparison.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape
    })

    prediction_data[model_name] = predictions

comparison_df = pd.DataFrame(comparison).sort_values("MAE")

comparison_df.to_csv(
    REPORT_DIR / "operational_pressure_model_comparison.csv",
    index=False
)

pd.DataFrame(prediction_data).to_csv(
    REPORT_DIR / "operational_pressure_model_predictions.csv",
    index=False
)

summary = [
    "OPERATIONAL PRESSURE MODEL TRAINING SUMMARY",
    "=" * 55,
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    f"Features: {len(FEATURES)}",
    f"Best validation model: {cv_df.sort_values('Average_MAE').iloc[0]['Model']}",
    "",
    "HELD-OUT TEST COMPARISON",
    comparison_df.to_string(index=False)
]

(REPORT_DIR / "operational_pressure_training_summary.txt").write_text(
    "\n".join(summary)
)

print("\nHELD-OUT TEST MODEL COMPARISON")
print(comparison_df.to_string(index=False))

print("\nTRAINING COMPLETED")
print("Models saved:", MODEL_DIR)
print("Reports saved:", REPORT_DIR)
