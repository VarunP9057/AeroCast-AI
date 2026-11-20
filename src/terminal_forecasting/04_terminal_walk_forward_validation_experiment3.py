
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
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/terminal_monthly_features.csv"

REPORT_DIR = (
    ROOT / "reports/terminal_forecasting_experiment3"
)

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
        Input(shape=(input_features,)),

        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.001)
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


def train_ann_with_full_fold(
    X_train,
    y_train,
    X_val,
    best_epochs,
    seed
):

    # Fit scalers using the complete outer training fold.
    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train
    )

    X_val_scaled = feature_scaler.transform(
        X_val
    )

    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    )

    # Rebuild the ANN and train on all outer training records.
    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(seed)

    model = build_ann(len(FEATURES))

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=best_epochs,
        batch_size=16,
        shuffle=False,
        verbose=0
    )

    scaled_predictions = model.predict(
        X_val_scaled,
        verbose=0
    )

    predictions = target_scaler.inverse_transform(
        scaled_predictions
    ).flatten()

    return predictions, len(history.history["loss"])


print("=" * 75)
print("TERMINAL-WISE WALK-FORWARD VALIDATION - EXPERIMENT 3")
print("=" * 75)

data = pd.read_csv(
    DATA_PATH,
    parse_dates=["YearMonth"]
)

all_fold_results = []
all_epoch_results = []

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

    # Preserve the original chronological development period.
    test_start = int(len(terminal_data) * 0.80)

    X_development = X.iloc[:test_start].reset_index(drop=True)
    y_development = y.iloc[:test_start].reset_index(drop=True)
    dates_development = dates.iloc[:test_start].reset_index(drop=True)

    splitter = TimeSeriesSplit(
        n_splits=4,
        test_size=24
    )

    terminal_fold_results = []

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(X_development),
        start=1
    ):

        print("\n" + "-" * 65)
        print(f"{terminal} - FOLD {fold}")
        print("-" * 65)

        X_train = X_development.iloc[train_indices]
        X_val = X_development.iloc[val_indices]

        y_train = y_development.iloc[train_indices]
        y_val = y_development.iloc[val_indices]

        val_dates = dates_development.iloc[val_indices]

        print("Outer training records:", len(X_train))
        print("Outer validation records:", len(X_val))

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

        terminal_fold_results.append({
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

        terminal_fold_results.append({
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

        terminal_fold_results.append({
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

        terminal_fold_results.append({
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
        # 5. ANN INNER VALIDATION
        # ------------------------------------------

        inner_split = int(len(X_train) * 0.80)

        X_fit = X_train.iloc[:inner_split]
        X_early_val = X_train.iloc[inner_split:]

        y_fit = y_train.iloc[:inner_split]
        y_early_val = y_train.iloc[inner_split:]

        inner_scaler = StandardScaler()

        X_fit_scaled = inner_scaler.fit_transform(X_fit)

        X_early_val_scaled = inner_scaler.transform(
            X_early_val
        )

        inner_target_scaler = StandardScaler()

        y_fit_scaled = inner_target_scaler.fit_transform(
            y_fit.to_numpy().reshape(-1, 1)
        )

        y_early_val_scaled = inner_target_scaler.transform(
            y_early_val.to_numpy().reshape(-1, 1)
        )

        seed = 42 + terminal_index * 100 + fold

        tf.keras.backend.clear_session()

        tf.keras.utils.set_random_seed(seed)

        selection_model = build_ann(len(FEATURES))

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

        selection_history = selection_model.fit(
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

        # Choose the epoch with the lowest inner validation loss.
        best_epochs = (
            int(
                np.argmin(
                    selection_history.history["val_loss"]
                )
            ) + 1
        )

        # ------------------------------------------
        # 6. ANN RETRAINING ON FULL OUTER FOLD
        # ------------------------------------------

        ann_predictions, final_epochs = train_ann_with_full_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            best_epochs=best_epochs,
            seed=seed + 1000
        )

        ann_metrics = calculate_metrics(
            y_val,
            ann_predictions
        )

        terminal_fold_results.append({
            "Terminal": terminal,
            "Fold": fold,
            "Model": "ANN",
            "Validation_Start": val_dates.min(),
            "Validation_End": val_dates.max(),
            "Training_Records": len(X_train),
            "Validation_Records": len(X_val),
            "ANN_Inner_Fit_Records": len(X_fit),
            "ANN_Inner_Validation_Records": len(X_early_val),
            "ANN_Selected_Epochs": best_epochs,
            "ANN_Final_Training_Records": len(X_train),
            "ANN_Final_Epochs": final_epochs,
            **ann_metrics
        })

        all_epoch_results.append({
            "Terminal": terminal,
            "Fold": fold,
            "Outer_Training_Records": len(X_train),
            "Inner_Fit_Records": len(X_fit),
            "Inner_Validation_Records": len(X_early_val),
            "Selected_Epochs": best_epochs,
            "Final_Training_Records": len(X_train),
            "Final_Epochs": final_epochs
        })

        print("ANN selected epochs:", best_epochs)
        print("ANN final training records:", len(X_train))
        print("ANN validation MAE:", round(ann_metrics["MAE"], 2))

    terminal_fold_df = pd.DataFrame(terminal_fold_results)

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
# SAVE FOLD RESULTS
# --------------------------------------------------

fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True
)

fold_path = (
    REPORT_DIR / "terminal_experiment3_fold_results.csv"
)

fold_results_df.to_csv(
    fold_path,
    index=False
)

# --------------------------------------------------
# SAVE ANN EPOCH SELECTION DETAILS
# --------------------------------------------------

epoch_results_df = pd.DataFrame(all_epoch_results)

epoch_path = (
    REPORT_DIR / "terminal_experiment3_ann_epochs.csv"
)

epoch_results_df.to_csv(
    epoch_path,
    index=False
)

# --------------------------------------------------
# CALCULATE VALIDATION SUMMARY
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
    REPORT_DIR / "terminal_experiment3_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

# --------------------------------------------------
# DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 75)
print("EXPERIMENT 3 - FINAL WALK-FORWARD VALIDATION SUMMARY")
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
        "\nBest validation model by MAE:",
        terminal_summary.iloc[0]["Model"]
    )

print("\nSaved reports:")
print(fold_path)
print(summary_path)
print(epoch_path)

print("\n" + "=" * 75)
print("EXPERIMENT 3 COMPLETED")
print("=" * 75)
