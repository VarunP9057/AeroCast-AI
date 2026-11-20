
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from keras.models import load_model


PROJECT_PATH = Path(__file__).resolve().parents[2]

MODEL_FOLDER = PROJECT_PATH / "models" / "ann"

MODEL_PATH = MODEL_FOLDER / "ann_passenger_model.keras"
SCALER_PATH = MODEL_FOLDER / "ann_scaler.pkl"
TARGET_SCALER_PATH = MODEL_FOLDER / "ann_target_scaler.pkl"


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
    "Rolling_Mean_6"
]


def load_forecasting_model():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        TARGET_SCALER_PATH
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {file_path}"
            )

    model = load_model(MODEL_PATH)
    input_scaler = joblib.load(SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    return model, input_scaler, target_scaler


def predict_passengers(features: dict):

    missing_features = [
        name for name in FEATURE_NAMES
        if name not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    try:
        input_values = [
            float(features[name])
            for name in FEATURE_NAMES
        ]
    except (TypeError, ValueError):
        raise ValueError(
            "All forecast features must be numeric."
        )

    input_array = np.array(input_values, dtype=float)

    if not np.isfinite(input_array).all():
        raise ValueError(
            "Forecast features must contain finite numbers."
        )

    if any(value < 0 for value in [
        features["Lag_1"],
        features["Lag_2"],
        features["Lag_3"],
        features["Lag_6"],
        features["Lag_12"],
        features["Rolling_Mean_3"],
        features["Rolling_Mean_6"]
    ]):
        raise ValueError(
            "Passenger counts and rolling averages cannot be negative."
        )

    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURE_NAMES
    )

    model, input_scaler, target_scaler = load_forecasting_model()

    scaled_input = input_scaler.transform(input_df)

    scaled_prediction = model.predict(
        scaled_input,
        verbose=0
    )

    prediction = target_scaler.inverse_transform(
        scaled_prediction
    )[0][0]

    prediction = max(float(prediction), 0.0)

    return round(prediction, 2)


def predict_custom_passengers(
    passenger_counts: list,
    target_year: int,
    target_month: int
):

    if len(passenger_counts) != 12:
        raise ValueError(
            "Exactly 12 monthly passenger counts are required."
        )

    counts = np.asarray(passenger_counts, dtype=float)

    if not np.isfinite(counts).all():
        raise ValueError(
            "Passenger counts must contain valid finite numbers."
        )

    if np.any(counts < 0):
        raise ValueError(
            "Passenger counts cannot be negative."
        )

    if not 1999 <= target_year <= 2100:
        raise ValueError("Target year must be between 1999 and 2100.")

    if not 1 <= target_month <= 12:
        raise ValueError("Target month must be between 1 and 12.")

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

        "Lag_1": float(counts[-1]),
        "Lag_2": float(counts[-2]),
        "Lag_3": float(counts[-3]),
        "Lag_6": float(counts[-6]),
        "Lag_12": float(counts[-12]),

        "Rolling_Mean_3": float(np.mean(counts[-3:])),
        "Rolling_Mean_6": float(np.mean(counts[-6:]))
    }

    prediction = predict_passengers(features)

    return {
        "predicted_passengers": prediction,
        "features_used": features
    }
