
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.keras.models import load_model


PROJECT_PATH = Path(__file__).resolve().parents[2]

MODEL_FOLDER = (
    PROJECT_PATH / "models" / "cargo_forecasting_experiment2"
)

MODEL_PATH = MODEL_FOLDER / "cargo_advanced_ann_model.keras"
SCALER_PATH = MODEL_FOLDER / "cargo_advanced_ann_scaler.pkl"
TARGET_SCALER_PATH = (
    MODEL_FOLDER / "cargo_advanced_ann_target_scaler.pkl"
)
FEATURE_PATH = (
    MODEL_FOLDER / "cargo_advanced_feature_names.pkl"
)

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "features"
    / "cargo_advanced_features.csv"
)


def load_cargo_model():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        TARGET_SCALER_PATH,
        FEATURE_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required cargo model file not found: {path}"
            )

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    if len(feature_names) != scaler.n_features_in_:
        raise ValueError("Cargo model feature count mismatch.")

    if target_scaler.n_features_in_ != 1:
        raise ValueError("Cargo target scaler must have one output.")

    return model, scaler, target_scaler, feature_names


def validate_cargo_features(features, feature_names):

    missing = [
        name for name in feature_names
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing cargo features: {missing}"
        )

    try:
        values = np.asarray(
            [float(features[name]) for name in feature_names],
            dtype=float
        )
    except (TypeError, ValueError):
        raise ValueError(
            "All cargo forecasting features must be numeric."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Cargo features must contain finite values."
        )

    return pd.DataFrame(
        [values],
        columns=feature_names
    )


def predict_cargo_weight(features):

    model, scaler, target_scaler, feature_names = (
        load_cargo_model()
    )

    input_df = validate_cargo_features(
        features,
        feature_names
    )

    scaled_input = scaler.transform(
    input_df.to_numpy()
)

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
            "Cargo model returned an invalid prediction."
        )

    return round(max(prediction, 0.0), 2)


def get_latest_cargo_features():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cargo feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Cargo feature dataset is empty.")

    row = df.iloc[-1]

    _, _, _, feature_names = load_cargo_model()

    features = {
        name: float(row[name])
        for name in feature_names
    }

    return str(row["YearMonth"])[:7], features


def predict_latest_cargo_weight():

    input_month, features = get_latest_cargo_features()

    prediction = predict_cargo_weight(features)

    return {
        "input_month": input_month,
        "predicted_cargo_weight_lbs": prediction,
        "unit": "lbs",
        "model_name": "Cargo Advanced ANN - Experiment 2",
        "status": "success"
    }

def predict_custom_cargo_weight(
    cargo_weights: list,
    target_year: int,
    target_month: int
):
    if len(cargo_weights) != 24:
        raise ValueError(
            "Exactly 24 monthly cargo weights are required."
        )

    weights = np.asarray(cargo_weights, dtype=float)

    if not np.isfinite(weights).all():
        raise ValueError(
            "Cargo weights must contain valid finite numbers."
        )

    if np.any(weights < 0):
        raise ValueError(
            "Cargo weights cannot be negative."
        )

    if not 1999 <= target_year <= 2100:
        raise ValueError(
            "Target year must be between 1999 and 2100."
        )

    if not 1 <= target_month <= 12:
        raise ValueError(
            "Target month must be between 1 and 12."
        )

    time_index = (
        (target_year - 1999) * 12
        + (target_month - 7)
    )

    if time_index < 24:
        raise ValueError(
            "The target month must have at least 24 months "
            "of historical data since July 1999."
        )

    previous_year_weight = weights[-13]

    if previous_year_weight == 0:
        raise ValueError(
            "The cargo weight 13 months before the target "
            "cannot be zero for percentage-change calculation."
        )

    previous_year_change = (
        weights[-1] - previous_year_weight
    )

    previous_year_change_pct = (
        previous_year_change / previous_year_weight
    ) * 100

    trend_slope_6 = np.polyfit(
        np.arange(6),
        weights[-6:],
        1
    )[0]

    trend_slope_12 = np.polyfit(
        np.arange(12),
        weights[-12:],
        1
    )[0]

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

        "Lag_1": float(weights[-1]),
        "Lag_2": float(weights[-2]),
        "Lag_3": float(weights[-3]),
        "Lag_6": float(weights[-6]),
        "Lag_12": float(weights[-12]),
        "Lag_18": float(weights[-18]),
        "Lag_24": float(weights[-24]),

        "Rolling_Mean_3": float(np.mean(weights[-3:])),
        "Rolling_Mean_6": float(np.mean(weights[-6:])),
        "Rolling_Mean_12": float(np.mean(weights[-12:])),
        "Rolling_Mean_24": float(np.mean(weights[-24:])),

        "Rolling_Std_3": float(np.std(weights[-3:], ddof=1)),
        "Rolling_Std_6": float(np.std(weights[-6:], ddof=1)),
        "Rolling_Std_12": float(np.std(weights[-12:], ddof=1)),

        "Previous_Year_Change": float(previous_year_change),
        "Previous_Year_Change_Pct": float(
            previous_year_change_pct
        ),

        "Trend_Slope_6": float(trend_slope_6),
        "Trend_Slope_12": float(trend_slope_12)
    }

    prediction = predict_cargo_weight(features)

    return {
        "predicted_cargo_weight_lbs": prediction,
        "input_month": f"{target_year:04d}-{target_month:02d}",
        "unit": "lbs",
        "model_name": "Cargo Advanced ANN - Experiment 2",
        "features_used": features,
        "status": "success"
    }
