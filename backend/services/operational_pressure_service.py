
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parents[2]

MODEL_FOLDER = (
    PROJECT_PATH / "models" / "operational_pressure_final"
)

MODEL_PATH = (
    MODEL_FOLDER / "operational_pressure_final_model.pkl"
)

FEATURE_PATH = (
    MODEL_FOLDER / "operational_pressure_final_features.pkl"
)

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "features"
    / "operational_pressure_features.csv"
)


def load_operational_pressure_model():

    required_files = [
        MODEL_PATH,
        FEATURE_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    if len(feature_names) != model.n_features_in_:
        raise ValueError(
            "Operational pressure feature count mismatch."
        )

    return model, feature_names


def validate_pressure_features(features, feature_names):

    missing = [
        name for name in feature_names
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing operational pressure features: {missing}"
        )

    try:
        values = np.asarray(
            [float(features[name]) for name in feature_names],
            dtype=float
        )
    except (TypeError, ValueError):
        raise ValueError(
            "All operational pressure features must be numeric."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Features must contain finite numeric values."
        )

    return pd.DataFrame(
        [values],
        columns=feature_names
    )


def predict_operational_pressure(features):

    model, feature_names = (
        load_operational_pressure_model()
    )

    input_df = validate_pressure_features(
        features,
        feature_names
    )

    prediction = model.predict(input_df)

    prediction = float(prediction[0])

    if not np.isfinite(prediction):
        raise ValueError(
            "Model returned an invalid prediction."
        )

    return round(prediction, 2)


def get_latest_pressure_features():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Operational pressure dataset is empty."
        )

    row = df.iloc[-1]

    _, feature_names = (
        load_operational_pressure_model()
    )

    features = {
        name: float(row[name])
        for name in feature_names
    }

    return str(row["YearMonth"])[:7], features


def predict_latest_operational_pressure():

    input_month, features = (
        get_latest_pressure_features()
    )

    prediction = predict_operational_pressure(features)

    return {
        "input_month": input_month,
        "predicted_operational_pressure_index": prediction,
        "actual_operational_pressure_index": round(
            float(
                pd.read_csv(DATA_PATH).iloc[-1][
                    "Operational Pressure Index"
                ]
            ),
            2
        ),
        "model_name": "Operational Pressure Linear Regression",
        "status": "success"
    }
