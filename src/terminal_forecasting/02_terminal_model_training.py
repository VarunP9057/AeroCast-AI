
from pathlib import Path
import pickle
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/terminal_monthly_features.csv"
MODEL_DIR = ROOT / "models/terminal_forecasting"
REPORT_DIR = ROOT / "reports/terminal_forecasting"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Passenger Count"

TERMINALS = [
    "International",
    "Terminal 1",
    "Terminal 2",
    "Terminal 3"
]

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
    "Rolling_Mean_12",
    "Rolling_Std_3",
    "Rolling_Std_6",
    "Rolling_Std_12"
]


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

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


def build_ann(input_features):

    model = Sequential([
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.001),
            input_shape=(input_features,)
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

    return model


print("=" * 75)
print("TERMINAL-WISE PASSENGER MODEL TRAINING")
print("=" * 75)

data = pd.read_csv(DATA_PATH, parse_dates=["YearMonth"])

print("\nFeature dataset shape:", data.shape)

all_results = []
all_predictions = []
all_training_histories = []

for terminal in TERMINALS:

    print("\n" + "=" * 75)
    print(f"TRAINING MODELS: {terminal.upper()}")
    print("=" * 75)

    terminal_data = (
        data[data["Terminal"] == terminal]
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )

    X = terminal_data[FEATURES].astype(float)
    y = terminal_data[TARGET].astype(float)

    dates = terminal_data["YearMonth"]

    split_index = int(len(terminal_data) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    train_dates = dates.iloc[:split_index]
    test_dates = dates.iloc[split_index:]

    print("Total records:", len(terminal_data))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training period:", train_dates.min(), "to", train_dates.max())
    print("Testing period:", test_dates.min(), "to", test_dates.max())

    terminal_slug = (
        terminal.lower()
        .replace(" ", "_")
    )

    terminal_model_dir = MODEL_DIR / terminal_slug
    terminal_model_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. SEASONAL NAIVE BASELINE
    # --------------------------------------------------

    print("\nTraining Seasonal Naive baseline...")

    seasonal_predictions = X_test["Lag_12"].values

    seasonal_metrics = calculate_metrics(
        y_test,
        seasonal_predictions
    )

    all_results.append({
        "Terminal": terminal,
        "Model": "Seasonal_Naive",
        **seasonal_metrics
    })

    for date, actual, predicted in zip(
        test_dates,
        y_test,
        seasonal_predictions
    ):
        all_predictions.append({
            "Terminal": terminal,
            "Model": "Seasonal_Naive",
            "YearMonth": date,
            "Actual": actual,
            "Predicted": predicted
        })

    # --------------------------------------------------
    # 2. LINEAR REGRESSION
    # --------------------------------------------------

    print("\nTraining Linear Regression...")

    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)

    linear_predictions = linear_model.predict(X_test)

    linear_metrics = calculate_metrics(
        y_test,
        linear_predictions
    )

    all_results.append({
        "Terminal": terminal,
        "Model": "Linear_Regression",
        **linear_metrics
    })

    with open(
        terminal_model_dir / "linear_regression.pkl",
        "wb"
    ) as file:
        pickle.dump(linear_model, file)

    for date, actual, predicted in zip(
        test_dates,
        y_test,
        linear_predictions
    ):
        all_predictions.append({
            "Terminal": terminal,
            "Model": "Linear_Regression",
            "YearMonth": date,
            "Actual": actual,
            "Predicted": predicted
        })

    # --------------------------------------------------
    # 3. RANDOM FOREST
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_metrics = calculate_metrics(
        y_test,
        rf_predictions
    )

    all_results.append({
        "Terminal": terminal,
        "Model": "Random_Forest",
        **rf_metrics
    })

    with open(
        terminal_model_dir / "random_forest.pkl",
        "wb"
    ) as file:
        pickle.dump(rf_model, file)

    for date, actual, predicted in zip(
        test_dates,
        y_test,
        rf_predictions
    ):
        all_predictions.append({
            "Terminal": terminal,
            "Model": "Random_Forest",
            "YearMonth": date,
            "Actual": actual,
            "Predicted": predicted
        })

    # --------------------------------------------------
    # 4. GRADIENT BOOSTING
    # --------------------------------------------------

    print("\nTraining Gradient Boosting...")

    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        min_samples_split=5,
        min_samples_leaf=3,
        loss="huber",
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    gb_predictions = gb_model.predict(X_test)

    gb_metrics = calculate_metrics(
        y_test,
        gb_predictions
    )

    all_results.append({
        "Terminal": terminal,
        "Model": "Gradient_Boosting",
        **gb_metrics
    })

    with open(
        terminal_model_dir / "gradient_boosting.pkl",
        "wb"
    ) as file:
        pickle.dump(gb_model, file)

    for date, actual, predicted in zip(
        test_dates,
        y_test,
        gb_predictions
    ):
        all_predictions.append({
            "Terminal": terminal,
            "Model": "Gradient_Boosting",
            "YearMonth": date,
            "Actual": actual,
            "Predicted": predicted
        })

    # --------------------------------------------------
    # 5. ARTIFICIAL NEURAL NETWORK
    # --------------------------------------------------

    print("\nTraining Artificial Neural Network...")

    validation_index = int(len(X_train) * 0.80)

    X_fit = X_train.iloc[:validation_index]
    X_val = X_train.iloc[validation_index:]

    y_fit = y_train.iloc[:validation_index]
    y_val = y_train.iloc[validation_index:]

    scaler = StandardScaler()

    X_fit_scaled = scaler.fit_transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    target_scaler = StandardScaler()

    y_fit_scaled = target_scaler.fit_transform(
        y_fit.to_numpy().reshape(-1, 1)
    )

    y_val_scaled = target_scaler.transform(
        y_val.to_numpy().reshape(-1, 1)
    )

    ann_model = build_ann(len(FEATURES))

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

    history = ann_model.fit(
        X_fit_scaled,
        y_fit_scaled,
        validation_data=(
            X_val_scaled,
            y_val_scaled
        ),
        epochs=300,
        batch_size=16,
        callbacks=[
            early_stopping,
            reduce_lr
        ],
        shuffle=False,
        verbose=0
    )

    ann_scaled_predictions = ann_model.predict(
        X_test_scaled,
        verbose=0
    )

    ann_predictions = target_scaler.inverse_transform(
        ann_scaled_predictions
    ).flatten()

    ann_metrics = calculate_metrics(
        y_test,
        ann_predictions
    )

    all_results.append({
        "Terminal": terminal,
        "Model": "ANN",
        **ann_metrics
    })

    ann_model.save(
        terminal_model_dir / "ann_model.keras"
    )

    with open(
        terminal_model_dir / "ann_scaler.pkl",
        "wb"
    ) as file:
        pickle.dump(scaler, file)

    with open(
        terminal_model_dir / "ann_target_scaler.pkl",
        "wb"
    ) as file:
        pickle.dump(target_scaler, file)

    with open(
        terminal_model_dir / "feature_names.pkl",
        "wb"
    ) as file:
        pickle.dump(FEATURES, file)

    history_df = pd.DataFrame(history.history)

    history_df["Terminal"] = terminal

    all_training_histories.append(history_df)

    for date, actual, predicted in zip(
        test_dates,
        y_test,
        ann_predictions
    ):
        all_predictions.append({
            "Terminal": terminal,
            "Model": "ANN",
            "YearMonth": date,
            "Actual": actual,
            "Predicted": predicted
        })

    # --------------------------------------------------
    # TERMINAL TRAINING SUMMARY
    # --------------------------------------------------

    terminal_results = pd.DataFrame([
        {
            "Model": "Seasonal_Naive",
            **seasonal_metrics
        },
        {
            "Model": "Linear_Regression",
            **linear_metrics
        },
        {
            "Model": "Random_Forest",
            **rf_metrics
        },
        {
            "Model": "Gradient_Boosting",
            **gb_metrics
        },
        {
            "Model": "ANN",
            **ann_metrics
        }
    ]).sort_values("MAE")

    print("\nMODEL COMPARISON:", terminal)

    print(
        terminal_results.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}"
        )
    )

    best_model = terminal_results.iloc[0]["Model"]

    metadata = {
        "terminal": terminal,
        "best_model_by_test_mae": best_model,
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "training_start": str(train_dates.min().date()),
        "training_end": str(train_dates.max().date()),
        "testing_start": str(test_dates.min().date()),
        "testing_end": str(test_dates.max().date()),
        "feature_count": len(FEATURES),
        "features": FEATURES
    }

    with open(
        terminal_model_dir / "training_metadata.json",
        "w"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nANN epochs:", len(history.history["loss"]))
    print("Best model by test MAE:", best_model)


# --------------------------------------------------
# SAVE CONSOLIDATED RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    ["Terminal", "MAE"]
).reset_index(drop=True)

predictions_df = pd.DataFrame(all_predictions)

history_df = pd.concat(
    all_training_histories,
    ignore_index=True
)

results_path = REPORT_DIR / "terminal_model_comparison.csv"
predictions_path = REPORT_DIR / "terminal_model_predictions.csv"
history_path = REPORT_DIR / "terminal_ann_training_history.csv"

results_df.to_csv(results_path, index=False)
predictions_df.to_csv(predictions_path, index=False)
history_df.to_csv(history_path, index=False)

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 75)
print("CONSOLIDATED TERMINAL MODEL COMPARISON")
print("=" * 75)

for terminal in TERMINALS:

    print(f"\n{terminal.upper()}")

    terminal_results = results_df[
        results_df["Terminal"] == terminal
    ].sort_values("MAE")

    print(
        terminal_results[
            ["Model", "MAE", "RMSE", "R2", "MAPE", "WAPE"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}"
        )
    )

print("\nSaved reports:")
print(results_path)
print(predictions_path)
print(history_path)

print("\n" + "=" * 75)
print("TERMINAL-WISE MODEL TRAINING COMPLETED")
print("=" * 75)
