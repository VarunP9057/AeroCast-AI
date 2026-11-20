
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.keras.models import load_model


PROJECT_PATH = Path(__file__).resolve().parents[2]

MODEL_FOLDER = PROJECT_PATH / "models" / "aircraft_forecasting"

MODEL_PATH = MODEL_FOLDER / "aircraft_ann_model.keras"
SCALER_PATH = MODEL_FOLDER / "aircraft_ann_scaler.pkl"
TARGET_SCALER_PATH = MODEL_FOLDER / "aircraft_ann_target_scaler.pkl"
FEATURE_PATH = MODEL_FOLDER / "aircraft_feature_names.pkl"

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "features"
    / "aircraft_monthly_features.csv"
)


def load_aircraft_model():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        TARGET_SCALER_PATH,
        FEATURE_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required aircraft model file not found: {path}"
            )

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    if len(feature_names) != scaler.n_features_in_:
        raise ValueError("Aircraft model feature count mismatch.")

    if list(scaler.feature_names_in_) != list(feature_names):
        raise ValueError("Aircraft scaler feature order mismatch.")

    return model, scaler, target_scaler, feature_names


def validate_aircraft_features(features, feature_names):

    missing = [
        name for name in feature_names
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing aircraft features: {missing}"
        )

    try:
        values = np.asarray(
            [float(features[name]) for name in feature_names],
            dtype=float
        )
    except (TypeError, ValueError):
        raise ValueError(
            "All aircraft forecasting features must be numeric."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Aircraft features must contain finite values."
        )

    if any(values[index] < 0 for index in range(5, 14)):
        raise ValueError(
            "Aircraft counts and rolling statistics cannot be negative."
        )

    return pd.DataFrame(
        [values],
        columns=feature_names
    )


def predict_aircraft_landings(features):

    model, scaler, target_scaler, feature_names = (
        load_aircraft_model()
    )

    input_df = validate_aircraft_features(
        features,
        feature_names
    )

    scaled_input = scaler.transform(input_df)

    scaled_prediction = model.predict(
        scaled_input,
        verbose=0
    )

    prediction = target_scaler.inverse_transform(
        scaled_prediction
    )[0][0]

    prediction = float(prediction)

    if not np.isfinite(prediction):
        raise ValueError(
            "Aircraft model returned an invalid prediction."
        )

    return round(max(prediction, 0.0), 2)


def get_latest_aircraft_features():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Aircraft feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Aircraft feature dataset is empty.")

    row = df.iloc[-1]

    _, _, _, feature_names = load_aircraft_model()

    features = {
        name: float(row[name])
        for name in feature_names
    }

    return str(row["YearMonth"]), features


def predict_latest_aircraft_landings():

    input_month, features = get_latest_aircraft_features()

    prediction = predict_aircraft_landings(features)

    return {
        "input_month": input_month,
        "predicted_landings": prediction,
        "model_name": "Original Aircraft ANN",
        "status": "success"
    }

def predict_custom_aircraft_landings(
    landing_counts: list,
    target_year: int,
    target_month: int
):
    if len(landing_counts) != 12:
        raise ValueError(
            "Exactly 12 monthly landing counts are required."
        )

    counts = np.asarray(landing_counts, dtype=float)

    if not np.isfinite(counts).all():
        raise ValueError(
            "Landing counts must contain valid finite numbers."
        )

    if np.any(counts < 0):
        raise ValueError(
            "Landing counts cannot be negative."
        )

    if not 1999 <= target_year <= 2100:
        raise ValueError(
            "Target year must be between 1999 and 2100."
        )

    if not 1 <= target_month <= 12:
        raise ValueError(
            "Target month must be between 1 and 12."
        )

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
        "Rolling_Mean_6": float(np.mean(counts[-6:])),

        "Rolling_Std_3": float(np.std(counts[-3:], ddof=1)),
        "Rolling_Std_6": float(np.std(counts[-6:], ddof=1))
    }

    prediction = predict_aircraft_landings(features)

    return {
        "predicted_landings": prediction,
        "input_month": f"{target_year:04d}-{target_month:02d}",
        "model_name": "Original Aircraft ANN",
        "features_used": features,
        "status": "success"
    }

