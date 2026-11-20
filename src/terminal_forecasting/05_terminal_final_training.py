
from pathlib import Path
import pickle
import json
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.keras.utils.set_random_seed(42)

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/terminal_monthly_features.csv"
MODEL_DIR = ROOT / "models/terminal_forecasting_final"
REPORT_DIR = ROOT / "reports/terminal_forecasting_final"

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

MODEL_SELECTION = {
    "International": {
        "model": "Random_Forest",
        "epochs": None
    },
    "Terminal 1": {
        "model": "ANN",
        "epochs": 18
    },
    "Terminal 2": {
        "model": "Linear_Regression",
        "epochs": None
    },
    "Terminal 3": {
        "model": "ANN",
        "epochs": 15
    }
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


print("=" * 75)
print("AEROCAST AI - FINAL TERMINAL MODEL TRAINING")
print("=" * 75)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_PATH}"
    )

data = pd.read_csv(
    DATA_PATH,
    parse_dates=["YearMonth"]
)

required_columns = [
    "YearMonth",
    "Terminal",
    TARGET
] + FEATURES

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if data[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values detected in required dataset columns."
    )

all_metadata = []

print("\nFeature dataset shape:", data.shape)

for terminal in TERMINALS:

    print("\n" + "=" * 75)
    print(f"FINAL TRAINING: {terminal.upper()}")
    print("=" * 75)

    terminal_data = (
        data[data["Terminal"] == terminal]
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )

    if terminal_data.empty:
        raise ValueError(
            f"No records found for {terminal}"
        )

    if terminal_data["YearMonth"].duplicated().any():
        raise ValueError(
            f"Duplicate monthly records found for {terminal}"
        )

    X = terminal_data[FEATURES].astype(float)
    y = terminal_data[TARGET].astype(float)

    terminal_slug = (
        terminal.lower().replace(" ", "_")
    )

    terminal_model_dir = MODEL_DIR / terminal_slug
    terminal_model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_model = MODEL_SELECTION[terminal]["model"]
    selected_epochs = MODEL_SELECTION[terminal]["epochs"]

    print("Selected model:", selected_model)
    print("Training records:", len(terminal_data))
    print("Features:", len(FEATURES))
    print("Training period:",
          terminal_data["YearMonth"].min().date(),
          "to",
          terminal_data["YearMonth"].max().date())

    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    if selected_model == "Random_Forest":

        print("\nTraining Random Forest...")

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y)

        model_path = (
            terminal_model_dir / "random_forest_final.pkl"
        )

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

    # --------------------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------------------

    elif selected_model == "Linear_Regression":

        print("\nTraining Linear Regression...")

        model = LinearRegression()

        model.fit(X, y)

        model_path = (
            terminal_model_dir / "linear_regression_final.pkl"
        )

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

    # --------------------------------------------------
    # ARTIFICIAL NEURAL NETWORK
    # --------------------------------------------------

    elif selected_model == "ANN":

        print("\nTraining Artificial Neural Network...")
        print("Selected epochs:", selected_epochs)

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        target_scaler = StandardScaler()

        y_scaled = target_scaler.fit_transform(
            y.to_numpy().reshape(-1, 1)
        )

        model = build_ann(len(FEATURES))

        history = model.fit(
            X_scaled,
            y_scaled,
            epochs=selected_epochs,
            batch_size=16,
            shuffle=False,
            verbose=1
        )

        model_path = (
            terminal_model_dir / "ann_final.keras"
        )

        model.save(model_path)

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

        history_df = pd.DataFrame(history.history)

        history_df.to_csv(
            terminal_model_dir / "ann_training_history.csv",
            index=False
        )

    else:
        raise ValueError(
            f"Unknown model selection: {selected_model}"
        )

    # --------------------------------------------------
    # SAVE FEATURE NAMES
    # --------------------------------------------------

    with open(
        terminal_model_dir / "feature_names.pkl",
        "wb"
    ) as file:
        pickle.dump(FEATURES, file)

    # --------------------------------------------------
    # SAVE MODEL METADATA
    # --------------------------------------------------

    metadata = {
        "terminal": terminal,
        "selected_model": selected_model,
        "training_records": len(terminal_data),
        "training_start": str(
            terminal_data["YearMonth"].min().date()
        ),
        "training_end": str(
            terminal_data["YearMonth"].max().date()
        ),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "selected_epochs": selected_epochs,
        "training_type": "Final deployment training",
        "model_file": model_path.name,
        "performance_note": (
            "Training completed using all available feature records. "
            "No independent test evaluation was performed in this script."
        )
    }

    with open(
        terminal_model_dir / "metadata.json",
        "w"
    ) as file:
        json.dump(metadata, file, indent=4)

    all_metadata.append({
        "Terminal": terminal,
        "Selected_Model": selected_model,
        "Training_Records": len(terminal_data),
        "Training_Start": terminal_data["YearMonth"].min(),
        "Training_End": terminal_data["YearMonth"].max(),
        "Features": len(FEATURES),
        "Selected_Epochs": selected_epochs,
        "Saved_Model": str(model_path)
    })

    print("\nModel saved:", model_path)
    print("Feature names saved.")
    print("Metadata saved.")
    print("Training completed successfully.")


# --------------------------------------------------
# SAVE CONSOLIDATED TRAINING SUMMARY
# --------------------------------------------------

summary_df = pd.DataFrame(all_metadata)

summary_path = (
    REPORT_DIR / "terminal_final_training_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 75)
print("FINAL TERMINAL MODEL TRAINING SUMMARY")
print("=" * 75)

print(
    summary_df.to_string(index=False)
)

print("\nSummary saved:", summary_path)

print("\n" + "=" * 75)
print("ALL FOUR FINAL TERMINAL MODELS TRAINED SUCCESSFULLY")
print("=" * 75)
