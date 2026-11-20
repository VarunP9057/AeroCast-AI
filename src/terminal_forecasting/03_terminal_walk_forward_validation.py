
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/terminal_monthly_features.csv"
REPORT_DIR = ROOT / "reports/terminal_forecasting"
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

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "MAPE": np.mean(
            np.abs((actual - predicted) / actual)
        ) * 100,
        "WAPE": (
            np.sum(np.abs(actual - predicted))
            / np.sum(np.abs(actual))
        ) * 100
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
print("TERMINAL-WISE WALK-FORWARD VALIDATION")
print("=" * 75)

data = pd.read_csv(
    DATA_PATH,
    parse_dates=["YearMonth"]
)

all_fold_results = []

for terminal_index, terminal in enumerate(TERMINALS):

    print("\n" + "=" * 75)
    print(f"VALIDATION: {terminal.upper()}")
    print("=" * 75)

    terminal_data = (
        data[data["Terminal"] == terminal]
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )

    X = terminal_data[FEATURES].astype(float)
    y = terminal_data[TARGET].astype(float)
    dates = terminal_data["YearMonth"]

    # Keep the same final test period as Experiment 1.
    test_start = int(len(terminal_data) * 0.80)

    X_development = X.iloc[:test_start].reset_index(drop=True)
    y_development = y.iloc[:test_start].reset_index(drop=True)
    dates_development = dates.iloc[:test_start].reset_index(drop=True)

    splitter = TimeSeriesSplit(
        n_splits=4,
        test_size=24
    )

    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(X_development),
        start=1
    ):

        print(f"\nFold {fold}")

        X_train = X_development.iloc[train_indices]
        X_val = X_development.iloc[val_indices]

        y_train = y_development.iloc[train_indices]
        y_val = y_development.iloc[val_indices]

        val_dates = dates_development.iloc[val_indices]

        print("Training records:", len(X_train))
        print("Validation records:", len(X_val))
        print(
            "Validation period:",
            val_dates.min().date(),
            "to",
            val_dates.max().date()
        )

        # ------------------------------------------
        # 1. SEASONAL NAIVE
        # ------------------------------------------

        seasonal_predictions = X_val["Lag_12"].values

        seasonal_metrics = calculate_metrics(
            y_val,
            seasonal_predictions
        )

        fold_results.append({
            "Terminal": terminal,
            "Fold": fold,
            "Model": "Seasonal_Naive",
            "Validation_Start": val_dates.min(),
            "Validation_End": val_dates.max(),
            "Training_Records": len(X_train),
            "Validation_Records": len(X_val),
            **seasonal_metrics
        })

        # ------------------------------------------
        # 2. LINEAR REGRESSION
        # ------------------------------------------

        linear_model = LinearRegression()

        linear_model.fit(X_train, y_train)

        linear_predictions = linear_model.predict(X_val)

        linear_metrics = calculate_metrics(
            y_val,
            linear_predictions
        )

        fold_results.append({
            "Terminal": terminal,
            "Fold": fold,
            "Model": "Linear_Regression",
            "Validation_Start": val_dates.min(),
            "Validation_End": val_dates.max(),
            "Training_Records": len(X_train),
            "Validation_Records": len(X_val),
            **linear_metrics
        })

        # ------------------------------------------
        # 3. RANDOM FOREST
        # ------------------------------------------

        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        rf_predictions = rf_model.predict(X_val)

        rf_metrics = calculate_metrics(
            y_val,
            rf_predictions
        )

        fold_results.append({
            "Terminal": terminal,
            "Fold": fold,
            "Model": "Random_Forest",
            "Validation_Start": val_dates.min(),
            "Validation_End": val_dates.max(),
            "Training_Records": len(X_train),
            "Validation_Records": len(X_val),
            **rf_metrics
        })

        # ------------------------------------------
        # 4. GRADIENT BOOSTING
        # ------------------------------------------

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

        gb_predictions = gb_model.predict(X_val)

        gb_metrics = calculate_metrics(
            y_val,
            gb_predictions
        )

        fold_results.append({
            "Terminal": terminal,
            "Fold": fold,
            "Model": "Gradient_Boosting",
            "Validation_Start": val_dates.min(),
            "Validation_End": val_dates.max(),
            "Training_Records": len(X_train),
            "Validation_Records": len(X_val),
            **gb_metrics
        })

        # ------------------------------------------
        # 5. ANN WITH INNER CHRONOLOGICAL VALIDATION
        # ------------------------------------------

        inner_split = int(len(X_train) * 0.80)

        X_fit = X_train.iloc[:inner_split]
        X_early_val = X_train.iloc[inner_split:]

        y_fit = y_train.iloc[:inner_split]
        y_early_val = y_train.iloc[inner_split:]

        scaler = StandardScaler()

        X_fit_scaled = scaler.fit_transform(X_fit)
        X_early_val_scaled = scaler.transform(X_early_val)
        X_val_scaled = scaler.transform(X_val)

        target_scaler = StandardScaler()

        y_fit_scaled = target_scaler.fit_transform(
            y_fit.to_numpy().reshape(-1, 1)
        )

        y_early_val_scaled = target_scaler.transform(
            y_early_val.to_numpy().reshape(-1, 1)
        )

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(
            42 + terminal_index * 100 + fold
        )

        ann_model = build_ann(len(FEATURES))

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
            validation_data=(
                X_early_val_scaled,
                y_early_val_scaled
            ),
            epochs=250,
            batch_size=16,
            callbacks=[
                early_stopping,
                reduce_lr
            ],
            shuffle=False,
            verbose=0
        )

        ann_scaled_predictions = ann_model.predict(
            X_val_scaled,
            verbose=0
        )

        ann_predictions = target_scaler.inverse_transform(
            ann_scaled_predictions
        ).flatten()

        ann_metrics = calculate_metrics(
            y_val,
            ann_predictions
        )

        fold_results.append({
            "Terminal": terminal,
            "Fold": fold,
            "Model": "ANN",
            "Validation_Start": val_dates.min(),
            "Validation_End": val_dates.max(),
            "Training_Records": len(X_train),
            "Validation_Records": len(X_val),
            "ANN_Fit_Records": len(X_fit),
            "ANN_Early_Stopping_Records": len(X_early_val),
            "ANN_Epochs": len(history.history["loss"]),
            **ann_metrics
        })

        print(
            "ANN validation MAE:",
            round(ann_metrics["MAE"], 2)
        )

    terminal_fold_df = pd.DataFrame(fold_results)

    all_fold_results.append(terminal_fold_df)

    print("\nFOLD RESULTS:", terminal)

    print(
        terminal_fold_df[
            ["Fold", "Model", "MAE", "RMSE", "R2", "MAPE", "WAPE"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}"
        )
    )


# --------------------------------------------------
# SAVE ALL FOLD RESULTS
# --------------------------------------------------

fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True
)

fold_path = REPORT_DIR / "terminal_walk_forward_fold_results.csv"

fold_results_df.to_csv(
    fold_path,
    index=False
)

# --------------------------------------------------
# CALCULATE AVERAGE VALIDATION PERFORMANCE
# --------------------------------------------------

summary_df = (
    fold_results_df
    .groupby(["Terminal", "Model"])
    .agg(
        Average_MAE=("MAE", "mean"),
        Std_MAE=("MAE", "std"),
        Average_RMSE=("RMSE", "mean"),
        Average_R2=("R2", "mean"),
        Average_MAPE=("MAPE", "mean"),
        Average_WAPE=("WAPE", "mean")
    )
    .reset_index()
)

summary_df = summary_df.sort_values(
    ["Terminal", "Average_MAE"]
).reset_index(drop=True)

summary_path = (
    REPORT_DIR / "terminal_walk_forward_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

# --------------------------------------------------
# DISPLAY FINAL VALIDATION RESULTS
# --------------------------------------------------

print("\n" + "=" * 75)
print("FINAL WALK-FORWARD VALIDATION SUMMARY")
print("=" * 75)

for terminal in TERMINALS:

    print(f"\n{terminal.upper()}")

    terminal_summary = summary_df[
        summary_df["Terminal"] == terminal
    ]

    print(
        terminal_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}"
        )
    )

    print(
        "\nBest validation model:",
        terminal_summary.iloc[0]["Model"]
    )

print("\nSaved reports:")
print(fold_path)
print(summary_path)

print("\n" + "=" * 75)
print("WALK-FORWARD VALIDATION COMPLETED")
print("=" * 75)
