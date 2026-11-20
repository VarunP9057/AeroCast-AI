
from pathlib import Path
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data/features/cargo_advanced_features.csv"

MODEL_DIR = ROOT / "models/cargo_forecasting_experiment2"
REPORT_DIR = ROOT / "reports/cargo_forecasting_experiment2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_START = pd.Timestamp("2021-06-01")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

TARGET = "Cargo Weight LBS"

print("\nDataset Shape:", df.shape)


# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

FEATURES = [
    "Year",
    "Month",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
    "Time_Index",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_6",
    "Lag_12",
    "Lag_18",
    "Lag_24",
    "Rolling_Mean_3",
    "Rolling_Mean_6",
    "Rolling_Mean_12",
    "Rolling_Mean_24",
    "Rolling_Std_3",
    "Rolling_Std_6",
    "Rolling_Std_12",
    "Previous_Year_Change",
    "Previous_Year_Change_Pct",
    "Trend_Slope_6",
    "Trend_Slope_12"
]

X = df[FEATURES].values
y = df[TARGET].values
dates = df["YearMonth"]


# --------------------------------------------------
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# --------------------------------------------------

train_mask = dates < TEST_START
test_mask = dates >= TEST_START

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

train_dates = dates[train_mask]
test_dates = dates[test_mask]

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nTraining Period:")
print(train_dates.min(), "to", train_dates.max())

print("\nTesting Period:")
print(test_dates.min(), "to", test_dates.max())

if len(X_test) != 63:
    raise ValueError(
        f"Expected 63 test records, but found {len(X_test)}."
    )


# --------------------------------------------------
# 5. MODEL FACTORY
# --------------------------------------------------

def create_model(model_name):

    if model_name == "Linear_Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

    if model_name == "Random_Forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if model_name == "Extra_Trees":
        return ExtraTreesRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if model_name == "Gradient_Boosting":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE
        )

    if model_name == "Hist_Gradient_Boosting":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.04,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        )

    raise ValueError(f"Unknown model: {model_name}")


MODEL_NAMES = [
    "Linear_Regression",
    "Random_Forest",
    "Extra_Trees",
    "Gradient_Boosting",
    "Hist_Gradient_Boosting"
]


# --------------------------------------------------
# 6. EVALUATION FUNCTION
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    wape = (
        np.sum(np.abs(actual - predicted)) /
        np.sum(np.abs(actual))
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape
    }


# --------------------------------------------------
# 7. TIME SERIES CROSS-VALIDATION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TIME SERIES CROSS-VALIDATION")
print("=" * 65)

tscv = TimeSeriesSplit(
    n_splits=4,
    test_size=24
)

cv_results = []

for model_name in MODEL_NAMES:

    for target_mode in ["Original", "Log1p"]:

        fold_mae = []

        for fold, (fit_index, val_index) in enumerate(
            tscv.split(X_train), start=1
        ):

            X_fit = X_train[fit_index]
            X_val = X_train[val_index]

            y_fit = y_train[fit_index]
            y_val = y_train[val_index]

            model = create_model(model_name)

            if target_mode == "Log1p":
                model.fit(X_fit, np.log1p(y_fit))
                predictions = np.expm1(
                    model.predict(X_val)
                )
            else:
                model.fit(X_fit, y_fit)
                predictions = model.predict(X_val)

            fold_score = mean_absolute_error(
                y_val, predictions
            )

            fold_mae.append(fold_score)

            print(
                f"{model_name} | {target_mode} | "
                f"Fold {fold} MAE: {fold_score:,.2f}"
            )

        cv_results.append({
            "Model": model_name,
            "Target_Transformation": target_mode,
            "Average_CV_MAE": np.mean(fold_mae),
            "CV_MAE_Std": np.std(fold_mae)
        })

cv_df = pd.DataFrame(cv_results)

cv_df = cv_df.sort_values(
    "Average_CV_MAE"
).reset_index(drop=True)

cv_df.to_csv(
    REPORT_DIR / "cargo_experiment2_cross_validation.csv",
    index=False
)

print("\nCross-Validation Ranking:")
print(cv_df.to_string(index=False))


# --------------------------------------------------
# 8. TRAIN AND TEST TRADITIONAL MODELS
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

comparison_results = []
prediction_data = {
    "YearMonth": test_dates.values,
    "Actual": y_test
}

cv_lookup = {
    (row["Model"], row["Target_Transformation"]):
    row["Average_CV_MAE"]
    for _, row in cv_df.iterrows()
}

for model_name in MODEL_NAMES:

    for target_mode in ["Original", "Log1p"]:

        model = create_model(model_name)

        if target_mode == "Log1p":
            model.fit(X_train, np.log1p(y_train))
            predictions = np.expm1(
                model.predict(X_test)
            )
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        predictions = np.maximum(predictions, 0)

        metrics = calculate_metrics(
            y_test, predictions
        )

        experiment_name = f"{model_name}_{target_mode}"

        comparison_results.append({
            "Model": experiment_name,
            "Base_Model": model_name,
            "Target_Transformation": target_mode,
            "CV_MAE": cv_lookup[
                (model_name, target_mode)
            ],
            **metrics
        })

        prediction_data[experiment_name] = predictions

        model_path = (
            MODEL_DIR / f"{experiment_name}.pkl"
        )

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        print(
            f"\n{experiment_name}\n"
            f"MAE: {metrics['MAE']:,.2f}\n"
            f"RMSE: {metrics['RMSE']:,.2f}\n"
            f"R2: {metrics['R2']:.4f}\n"
            f"MAPE: {metrics['MAPE']:.2f}%\n"
            f"WAPE: {metrics['WAPE']:.2f}%"
        )


# --------------------------------------------------
# 9. ANN MODEL WITH LEAKAGE-SAFE VALIDATION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING ADVANCED ANN")
print("=" * 65)

validation_split_index = int(len(X_train) * 0.80)

X_ann_fit = X_train[:validation_split_index]
X_ann_val = X_train[validation_split_index:]

y_ann_fit = y_train[:validation_split_index]
y_ann_val = y_train[validation_split_index:]

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_ann_fit_scaled = X_scaler.fit_transform(X_ann_fit)
X_ann_val_scaled = X_scaler.transform(X_ann_val)
X_test_scaled = X_scaler.transform(X_test)

y_ann_fit_scaled = y_scaler.fit_transform(
    y_ann_fit.reshape(-1, 1)
)

y_ann_val_scaled = y_scaler.transform(
    y_ann_val.reshape(-1, 1)
)

ann_model = Sequential([
    Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(0.001),
        input_shape=(X_ann_fit_scaled.shape[1],)
    ),
    Dropout(0.20),

    Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),
    Dropout(0.10),

    Dense(16, activation="relu"),

    Dense(1)
])

ann_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="mse",
    metrics=["mae"]
)

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
    X_ann_fit_scaled,
    y_ann_fit_scaled,
    validation_data=(
        X_ann_val_scaled,
        y_ann_val_scaled
    ),
    epochs=300,
    batch_size=16,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    shuffle=False,
    verbose=1
)

ann_predictions_scaled = ann_model.predict(
    X_test_scaled,
    verbose=0
)

ann_predictions = y_scaler.inverse_transform(
    ann_predictions_scaled
).flatten()

ann_predictions = np.maximum(
    ann_predictions, 0
)

ann_metrics = calculate_metrics(
    y_test, ann_predictions
)

comparison_results.append({
    "Model": "Advanced_ANN",
    "Base_Model": "ANN",
    "Target_Transformation": "StandardScaler",
    "CV_MAE": np.nan,
    **ann_metrics
})

prediction_data["Advanced_ANN"] = ann_predictions

ann_model.save(
    MODEL_DIR / "cargo_advanced_ann_model.keras"
)

with open(
    MODEL_DIR / "cargo_advanced_ann_scaler.pkl", "wb"
) as file:
    pickle.dump(X_scaler, file)

with open(
    MODEL_DIR / "cargo_advanced_ann_target_scaler.pkl", "wb"
) as file:
    pickle.dump(y_scaler, file)

with open(
    MODEL_DIR / "cargo_advanced_feature_names.pkl", "wb"
) as file:
    pickle.dump(FEATURES, file)

pd.DataFrame(history.history).to_csv(
    REPORT_DIR / "cargo_advanced_ann_training_history.csv",
    index=False
)


# --------------------------------------------------
# 10. SAVE FINAL RESULTS
# --------------------------------------------------

comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(
    "MAE"
).reset_index(drop=True)

predictions_df = pd.DataFrame(prediction_data)

comparison_df.to_csv(
    REPORT_DIR / "cargo_experiment2_model_comparison.csv",
    index=False
)

predictions_df.to_csv(
    REPORT_DIR / "cargo_experiment2_model_predictions.csv",
    index=False
)


# --------------------------------------------------
# 11. SAVE TRAINING SUMMARY
# --------------------------------------------------

best_model = comparison_df.iloc[0]

summary = f"""
CARGO FORECASTING EXPERIMENT 2
=======================================

Dataset:
{DATA_FILE}

Total Records: {len(df)}
Input Features: {len(FEATURES)}

Training Records: {len(X_train)}
Testing Records: {len(X_test)}

Training Period:
{train_dates.min()} to {train_dates.max()}

Testing Period:
{test_dates.min()} to {test_dates.max()}

Best Test Model: {best_model['Model']}

Best Test MAE: {best_model['MAE']:.2f}
Best Test RMSE: {best_model['RMSE']:.2f}
Best Test R2: {best_model['R2']:.4f}
Best Test MAPE: {best_model['MAPE']:.2f}%
Best Test WAPE: {best_model['WAPE']:.2f}%

Note:
The test results are held-out evaluation results.
CV MAE is used as the training-period model-selection
criterion for traditional models.
The ANN is reported separately because it was not
included in the traditional-model cross-validation.
"""

with open(
    REPORT_DIR / "cargo_experiment2_training_summary.txt",
    "w"
) as file:
    file.write(summary)


# --------------------------------------------------
# 12. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 65)
print("CARGO EXPERIMENT 2 COMPLETED")
print("=" * 65)

print("\nFinal Model Ranking:")
print(
    comparison_df[
        ["Model", "MAE", "RMSE", "R2", "MAPE", "WAPE"]
    ].to_string(index=False)
)

print("\nBest Test Model:", best_model["Model"])

print("\nSaved Models:")
print(MODEL_DIR)

print("\nSaved Reports:")
print(REPORT_DIR)
