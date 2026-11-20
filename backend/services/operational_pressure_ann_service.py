from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "operational_pressure"
DATA_FILE = BASE_DIR / "data" / "features" / "operational_pressure_features.csv"


def predict_latest_operational_pressure_ann():
    required_files = [
        DATA_FILE,
        MODEL_DIR / "operational_pressure_feature_names.pkl",
        MODEL_DIR / "operational_pressure_x_scaler.pkl",
        MODEL_DIR / "operational_pressure_y_scaler.pkl",
        MODEL_DIR / "operational_pressure_ann.keras",
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required ANN file not found: {file_path}")

    data = pd.read_csv(DATA_FILE)

    if data.empty:
        raise ValueError("Operational pressure feature dataset is empty.")

    feature_names = joblib.load(
        MODEL_DIR / "operational_pressure_feature_names.pkl"
    )
    scaler = joblib.load(
        MODEL_DIR / "operational_pressure_x_scaler.pkl"
    )
    target_scaler = joblib.load(
        MODEL_DIR / "operational_pressure_y_scaler.pkl"
    )
    model = tf.keras.models.load_model(
        MODEL_DIR / "operational_pressure_ann.keras",
        compile=False
    )

    latest_row = data.iloc[-1]

    missing_features = [
        name for name in feature_names if name not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing operational pressure features: {missing_features}"
        )

    input_data = pd.DataFrame(
        [[float(latest_row[name]) for name in feature_names]],
        columns=feature_names
    )

    scaled_input = scaler.transform(input_data)
    scaled_prediction = model.predict(scaled_input, verbose=0)
    prediction = target_scaler.inverse_transform(
        np.asarray(scaled_prediction).reshape(-1, 1)
    )[0][0]

    year = int(latest_row["Year"])
    month = int(latest_row["Month"])

    result = {
        "input_month": f"{year:04d}-{month:02d}",
        "predicted_operational_pressure_index": round(float(prediction), 2),
        "model_name": "Operational Pressure ANN",
        "status": "success"
    }

    actual_column = next(
        (
            column for column in [
                "Operational_Pressure_Index",
                "Operational Pressure Index",
                "operational_pressure_index"
            ]
            if column in data.columns
        ),
        None
    )

    if actual_column:
        result["actual_operational_pressure_index"] = round(
            float(latest_row[actual_column]), 2
        )

    return result

def predict_custom_operational_pressure_ann(
    passenger_counts: list,
    landing_counts: list,
    cargo_weights: list,
    target_year: int,
    target_month: int
):
    if (
        len(passenger_counts) != 12
        or len(landing_counts) != 12
        or len(cargo_weights) != 12
    ):
        raise ValueError(
            "Exactly 12 monthly values are required for each indicator."
        )

    passengers = np.asarray(passenger_counts, dtype=float)
    aircraft = np.asarray(landing_counts, dtype=float)
    cargo = np.asarray(cargo_weights, dtype=float)

    for name, values in [
        ("Passenger counts", passengers),
        ("Aircraft landing counts", aircraft),
        ("Cargo weights", cargo)
    ]:
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain valid finite numbers.")

        if np.any(values < 0):
            raise ValueError(f"{name} cannot contain negative values.")

    if not 1999 <= target_year <= 2100:
        raise ValueError("Target year must be between 1999 and 2100.")

    if not 1 <= target_month <= 12:
        raise ValueError("Target month must be between 1 and 12.")

    time_index = (
        (target_year - 1999) * 12
        + (target_month - 7)
    )

    if time_index < 12:
        raise ValueError(
            "The target month must have at least 12 months "
            "of historical data since July 1999."
        )

    passenger_mean = 3339107.7300380226
    passenger_std = 1010550.9588100557

    aircraft_mean = 14893.589353612168
    aircraft_std = 2792.7445648987873

    cargo_mean = 96560545.51330799
    cargo_std = 24209892.72780527

    training_score_min = -2.935505574594652
    training_score_max = 1.8201583879313545

    passenger_normalized = (
        passengers - passenger_mean
    ) / passenger_std

    aircraft_normalized = (
        aircraft - aircraft_mean
    ) / aircraft_std

    cargo_normalized = (
        cargo - cargo_mean
    ) / cargo_std

    combined_score = (
        passenger_normalized
        + aircraft_normalized
        + cargo_normalized
    ) / 3

    pressure_history = (
        (combined_score - training_score_min)
        / (training_score_max - training_score_min)
    ) * 100

    features = {
        "Year": int(target_year),
        "Month": int(target_month),
        "Quarter": int((target_month - 1) // 3 + 1),

        "Month_Sin": float(
            np.sin(2 * np.pi * target_month / 12)
        ),

        "Month_Cos": float(
            np.cos(2 * np.pi * target_month / 12)
        ),

        "Time_Index": int(time_index),

        "Pressure_Lag_1": float(pressure_history[-1]),
        "Pressure_Lag_2": float(pressure_history[-2]),
        "Pressure_Lag_3": float(pressure_history[-3]),
        "Pressure_Lag_6": float(pressure_history[-6]),
        "Pressure_Lag_12": float(pressure_history[-12]),

        "Pressure_Rolling_Mean_3": float(
            np.mean(pressure_history[-3:])
        ),

        "Pressure_Rolling_Mean_6": float(
            np.mean(pressure_history[-6:])
        ),

        "Pressure_Rolling_Mean_12": float(
            np.mean(pressure_history[-12:])
        ),

        "Pressure_Rolling_Std_3": float(
            np.std(pressure_history[-3:], ddof=1)
        ),

        "Pressure_Rolling_Std_6": float(
            np.std(pressure_history[-6:], ddof=1)
        ),

        "Pressure_Rolling_Std_12": float(
            np.std(pressure_history[-12:], ddof=1)
        ),

        "Passenger_Lag_1": float(passengers[-1]),
        "Passenger_Lag_12": float(passengers[-12]),

        "Aircraft_Lag_1": float(aircraft[-1]),
        "Aircraft_Lag_12": float(aircraft[-12]),

        "Cargo_Lag_1": float(cargo[-1]),
        "Cargo_Lag_12": float(cargo[-12])
    }

    required_files = [
        MODEL_DIR / "operational_pressure_feature_names.pkl",
        MODEL_DIR / "operational_pressure_x_scaler.pkl",
        MODEL_DIR / "operational_pressure_y_scaler.pkl",
        MODEL_DIR / "operational_pressure_ann.keras"
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required ANN file not found: {file_path}"
            )

    feature_names = joblib.load(
        MODEL_DIR / "operational_pressure_feature_names.pkl"
    )

    if len(feature_names) != 23:
        raise ValueError("Operational Pressure ANN must have 23 features.")

    scaler = joblib.load(
        MODEL_DIR / "operational_pressure_x_scaler.pkl"
    )

    target_scaler = joblib.load(
        MODEL_DIR / "operational_pressure_y_scaler.pkl"
    )

    model = tf.keras.models.load_model(
        MODEL_DIR / "operational_pressure_ann.keras",
        compile=False
    )

    input_data = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    scaled_input = scaler.transform(input_data)

    scaled_prediction = model.predict(
        scaled_input,
        verbose=0
    )

    prediction = target_scaler.inverse_transform(
        np.asarray(scaled_prediction).reshape(-1, 1)
    )[0][0]

    if not np.isfinite(prediction):
        raise ValueError(
            "Operational Pressure ANN returned an invalid prediction."
        )

    return {
        "input_month": f"{target_year:04d}-{target_month:02d}",
        "predicted_operational_pressure_index": round(
            float(prediction), 2
        ),
        "model_name": "Operational Pressure ANN",
        "features_used": features,
        "pressure_history": [
            round(float(value), 4)
            for value in pressure_history
        ],
        "status": "success"
    }

