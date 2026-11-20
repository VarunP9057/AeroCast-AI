
import pickle
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parents[2]

MODEL_FOLDER = PROJECT_PATH / "models" / "terminal_forecasting_final"

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "features"
    / "terminal_monthly_features.csv"
)

FEATURE_NAMES = [
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

TERMINAL_CONFIG = {
    "International": {
        "folder": "international",
        "model_type": "Random_Forest",
        "model_file": "random_forest_final.pkl"
    },
    "Terminal 1": {
        "folder": "terminal_1",
        "model_type": "ANN",
        "model_file": "ann_final.keras"
    },
    "Terminal 2": {
        "folder": "terminal_2",
        "model_type": "Linear_Regression",
        "model_file": "linear_regression_final.pkl"
    },
    "Terminal 3": {
        "folder": "terminal_3",
        "model_type": "ANN",
        "model_file": "ann_final.keras"
    }
}


def load_pickle(path):

    with open(path, "rb") as file:
        return pickle.load(file)


def load_terminal_model(terminal):

    if terminal not in TERMINAL_CONFIG:
        raise ValueError(
            f"Unsupported terminal: {terminal}"
        )

    config = TERMINAL_CONFIG[terminal]

    model_folder = MODEL_FOLDER / config["folder"]

    metadata_path = model_folder / "metadata.json"
    feature_path = model_folder / "feature_names.pkl"
    model_path = model_folder / config["model_file"]

    for path in [metadata_path, feature_path, model_path]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    features = load_pickle(feature_path)

    if features != FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch for {terminal}"
        )

    if metadata["selected_model"] != config["model_type"]:
        raise ValueError(
            f"Model metadata mismatch for {terminal}"
        )

    if config["model_type"] == "ANN":

        scaler_path = model_folder / "ann_scaler.pkl"
        target_scaler_path = (
            model_folder / "ann_target_scaler.pkl"
        )

        if not scaler_path.exists() or not target_scaler_path.exists():
            raise FileNotFoundError(
                f"ANN scalers missing for {terminal}"
            )

        model = tf.keras.models.load_model(model_path)

        scaler = load_pickle(scaler_path)
        target_scaler = load_pickle(target_scaler_path)

        return model, scaler, target_scaler

    model = load_pickle(model_path)

    return model, None, None


def validate_features(features):

    missing = [
        name
        for name in FEATURE_NAMES
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    values = np.asarray(
        [features[name] for name in FEATURE_NAMES],
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "All features must contain finite numeric values."
        )

    if any(values[index] < 0 for index in range(5, 16)):
        raise ValueError(
            "Passenger lags and rolling statistics cannot be negative."
        )

    return pd.DataFrame(
        [values],
        columns=FEATURE_NAMES
    )


def predict_terminal_passengers(terminal, features):

    if terminal not in TERMINAL_CONFIG:
        raise ValueError(
            "Choose International, Terminal 1, Terminal 2 or Terminal 3."
        )

    input_df = validate_features(features)

    model, scaler, target_scaler = load_terminal_model(
        terminal
    )

    if TERMINAL_CONFIG[terminal]["model_type"] == "ANN":

        scaled_input = scaler.transform(input_df)

        scaled_prediction = model.predict(
            scaled_input,
            verbose=0
        )

        prediction = target_scaler.inverse_transform(
            scaled_prediction
        )[0][0]

    else:

        prediction = model.predict(input_df)[0]

    prediction = float(prediction)

    if not np.isfinite(prediction):
        raise ValueError(
            "Model returned an invalid prediction."
        )

    return {
        "terminal": terminal,
        "predicted_passengers": round(
            max(prediction, 0.0),
            2
        ),
        "model_name": TERMINAL_CONFIG[terminal]["model_type"],
        "status": "success"
    }


def get_latest_terminal_features(terminal):

    if terminal not in TERMINAL_CONFIG:
        raise ValueError(
            f"Unsupported terminal: {terminal}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Terminal dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["YearMonth"]
    )

    terminal_df = (
        df[df["Terminal"] == terminal]
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )

    if terminal_df.empty:
        raise ValueError(
            f"No historical data found for {terminal}"
        )

    row = terminal_df.iloc[-1]

    return {
        name: float(row[name])
        for name in FEATURE_NAMES
    }


def predict_latest_terminal_passengers(terminal):

    features = get_latest_terminal_features(terminal)

    result = predict_terminal_passengers(
        terminal,
        features
    )

    result["input_month"] = (
        pd.Timestamp(
            pd.read_csv(DATA_PATH).loc[
                pd.read_csv(DATA_PATH)["Terminal"] == terminal,
                "YearMonth"
            ].iloc[-1]
        ).strftime("%Y-%m")
    )

    return result


def get_all_terminal_predictions():

    results = []

    for terminal in TERMINAL_CONFIG:

        results.append(
            predict_latest_terminal_passengers(terminal)
        )

    total_prediction = sum(
        result["predicted_passengers"]
        for result in results
    )

    return {
        "terminal_predictions": results,
        "combined_predicted_passengers": round(
            total_prediction,
            2
        ),
        "status": "success"
    }

def predict_custom_terminal_passengers(
    terminal,
    passenger_counts,
    target_year,
    target_month
):
    if terminal not in ["Terminal 1", "Terminal 3"]:
        raise ValueError(
            "Custom ANN prediction is available only for Terminal 1 and Terminal 3."
        )

    counts = np.asarray(passenger_counts, dtype=float)

    if len(counts) != 12:
        raise ValueError("Exactly 12 monthly passenger counts are required.")

    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Passenger counts must be finite and non-negative.")

    target_year = int(target_year)
    target_month = int(target_month)

    if target_year < 1:
        raise ValueError("Enter a valid target year.")

    if target_month < 1 or target_month > 12:
        raise ValueError("Target month must be between 1 and 12.")

    features = {
        "Year": target_year,
        "Month": target_month,
        "Quarter": (target_month - 1) // 3 + 1,
        "Month_Sin": np.sin(2 * np.pi * target_month / 12),
        "Month_Cos": np.cos(2 * np.pi * target_month / 12),
        "Lag_1": counts[-1],
        "Lag_2": counts[-2],
        "Lag_3": counts[-3],
        "Lag_6": counts[-6],
        "Lag_12": counts[-12],
        "Rolling_Mean_3": np.mean(counts[-3:]),
        "Rolling_Mean_6": np.mean(counts[-6:]),
        "Rolling_Mean_12": np.mean(counts[-12:]),
        "Rolling_Std_3": np.std(counts[-3:], ddof=1),
        "Rolling_Std_6": np.std(counts[-6:], ddof=1),
        "Rolling_Std_12": np.std(counts[-12:], ddof=1)
    }

    result = predict_terminal_passengers(terminal, features)

    result["target_month"] = f"{target_year}-{target_month:02d}"
    result["input_history_months"] = 12
    result["model_name"] = f"{terminal} ANN"

    return result

