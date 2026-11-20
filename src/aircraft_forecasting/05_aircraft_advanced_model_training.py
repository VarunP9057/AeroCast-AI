
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT / "data" / "features"
    / "aircraft_advanced_features.csv"
)

ORIGINAL_PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "aircraft_forecasting"
    / "aircraft_model_predictions.csv"
)

MODEL_FOLDER = (
    PROJECT_ROOT / "models" / "aircraft_forecasting_experiment2"
)

REPORT_FOLDER = (
    PROJECT_ROOT / "reports" / "aircraft_forecasting_experiment2"
)

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

TEST_START = "2021-06"

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("\n========== AEROCAST AI ==========")
print("AIRCRAFT ADVANCED MODEL TRAINING")
print("EXPERIMENT 2")
print("================================")

# 1. Load dataset

df = pd.read_csv(FEATURE_FILE)

df["YearMonth"] = pd.to_datetime(
    df["YearMonth"], format="%Y-%m"
)

df = df.sort_values("YearMonth").reset_index(drop=True)

# 2. Select features

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
    "Lag_9",
    "Lag_12",
    "Lag_18",
    "Lag_24",
    "YoY_Change",
    "YoY_Growth_Percent",
    "Rolling_Mean_3",
    "Rolling_Mean_6",
    "Rolling_Mean_9",
    "Rolling_Mean_12",
    "Rolling_Std_3",
    "Rolling_Std_6",
    "Rolling_Std_12",
    "Recent_Trend"
]

TARGET = "Landing Count"

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)

# 3. Chronological train-test split

train_mask = df["YearMonth"] < pd.Timestamp(TEST_START)
test_mask = df["YearMonth"] >= pd.Timestamp(TEST_START)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

test_dates = df.loc[test_mask, "YearMonth"].dt.strftime("%Y-%m")

print("\nDataset Shape:", df.shape)
print("Number of Features:", len(FEATURES))

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print(
    "Training Period:",
    df.loc[train_mask, "YearMonth"].min().strftime("%Y-%m"),
    "to",
    df.loc[train_mask, "YearMonth"].max().strftime("%Y-%m")
)

print(
    "Testing Period:",
    df.loc[test_mask, "YearMonth"].min().strftime("%Y-%m"),
    "to",
    df.loc[test_mask, "YearMonth"].max().strftime("%Y-%m")
)

if len(X_test) != 63:
    raise ValueError(
        "Expected exactly 63 test records. Check the dataset dates."
    )

if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Missing values found in model data.")

# 4. Evaluation function

def evaluate_model(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(np.abs(actual))
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "WAPE": wape
    }

results = []
prediction_data = pd.DataFrame({
    "YearMonth": test_dates.values,
    "Actual": y_test.values
})

# 5. Seasonal naive baseline

seasonal_predictions = df.loc[
    test_mask, "Lag_12"
].values

prediction_data["Seasonal_Naive"] = seasonal_predictions

results.append({
    "Model": "Seasonal_Naive",
    **evaluate_model(y_test.values, seasonal_predictions)
})

# 6. Linear Regression

linear_scaler = StandardScaler()

X_train_linear = linear_scaler.fit_transform(X_train)
X_test_linear = linear_scaler.transform(X_test)

linear_model = LinearRegression()

linear_model.fit(
    X_train_linear,
    y_train
)

linear_predictions = linear_model.predict(
    X_test_linear
)

prediction_data["Linear_Regression"] = linear_predictions

results.append({
    "Model": "Linear_Regression",
    **evaluate_model(y_test.values, linear_predictions)
})

# 7. Random Forest

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

prediction_data["Random_Forest"] = rf_predictions

results.append({
    "Model": "Random_Forest",
    **evaluate_model(y_test.values, rf_predictions)
})

# 8. ANN preprocessing

validation_size = int(len(X_train) * 0.20)

fit_size = len(X_train) - validation_size

X_fit = X_train.iloc[:fit_size]
X_validation = X_train.iloc[fit_size:]

y_fit = y_train.iloc[:fit_size]
y_validation = y_train.iloc[fit_size:]

ann_scaler = StandardScaler()
ann_target_scaler = StandardScaler()

X_fit_scaled = ann_scaler.fit_transform(X_fit)

X_validation_scaled = ann_scaler.transform(
    X_validation
)

X_test_scaled = ann_scaler.transform(X_test)

y_fit_scaled = ann_target_scaler.fit_transform(
    y_fit.values.reshape(-1, 1)
)

y_validation_scaled = ann_target_scaler.transform(
    y_validation.values.reshape(-1, 1)
)

print("\nANN Training Records:", len(X_fit))
print("ANN Validation Records:", len(X_validation))

# 9. Build advanced ANN

ann_model = Sequential([
    Input(shape=(len(FEATURES),)),

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

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])

ann_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

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

# 10. Train ANN

history = ann_model.fit(
    X_fit_scaled,
    y_fit_scaled,
    validation_data=(
        X_validation_scaled,
        y_validation_scaled
    ),
    epochs=250,
    batch_size=16,
    shuffle=False,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)

# 11. ANN predictions

ann_scaled_predictions = ann_model.predict(
    X_test_scaled,
    verbose=0
)

ann_predictions = ann_target_scaler.inverse_transform(
    ann_scaled_predictions
).flatten()

prediction_data["ANN_Experiment_2"] = ann_predictions

results.append({
    "Model": "ANN_Experiment_2",
    **evaluate_model(y_test.values, ann_predictions)
})

# 12. Load original ANN predictions

original = pd.read_csv(ORIGINAL_PREDICTIONS_FILE)

original["YearMonth"] = original["YearMonth"].astype(str)

prediction_data["YearMonth"] = (
    prediction_data["YearMonth"].astype(str)
)

original_ann = original[
    ["YearMonth", "ANN"]
].rename(
    columns={"ANN": "Original_ANN"}
)

prediction_data = prediction_data.merge(
    original_ann,
    on="YearMonth",
    how="left",
    validate="one_to_one"
)

if prediction_data["Original_ANN"].isnull().any():
    raise ValueError(
        "Original ANN predictions do not cover all test months."
    )

results.append({
    "Model": "Original_ANN",
    **evaluate_model(
        prediction_data["Actual"].values,
        prediction_data["Original_ANN"].values
    )
})

# 13. Model comparison

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "MAE"
).reset_index(drop=True)

comparison.to_csv(
    REPORT_FOLDER / "aircraft_experiment2_model_comparison.csv",
    index=False
)

# 14. Save predictions

prediction_data.to_csv(
    REPORT_FOLDER / "aircraft_experiment2_predictions.csv",
    index=False
)

# 15. Save ANN training history

pd.DataFrame(history.history).to_csv(
    REPORT_FOLDER / "aircraft_experiment2_training_history.csv",
    index=False
)

# 16. Save models and scalers

ann_model.save(
    MODEL_FOLDER / "aircraft_advanced_ann.keras"
)

joblib.dump(
    ann_scaler,
    MODEL_FOLDER / "aircraft_advanced_ann_scaler.pkl"
)

joblib.dump(
    ann_target_scaler,
    MODEL_FOLDER / "aircraft_advanced_target_scaler.pkl"
)

joblib.dump(
    linear_model,
    MODEL_FOLDER / "aircraft_advanced_linear_regression.pkl"
)

joblib.dump(
    linear_scaler,
    MODEL_FOLDER / "aircraft_advanced_linear_scaler.pkl"
)

joblib.dump(
    rf_model,
    MODEL_FOLDER / "aircraft_advanced_random_forest.pkl"
)

joblib.dump(
    FEATURES,
    MODEL_FOLDER / "aircraft_advanced_feature_names.pkl"
)

# 17. Analyze the four important months

important_months = [
    "2023-07",
    "2024-06",
    "2025-02",
    "2024-02"
]

selected = prediction_data[
    prediction_data["YearMonth"].isin(important_months)
].copy()

selected["Original_ANN_Error"] = (
    selected["Actual"] - selected["Original_ANN"]
).abs()

selected["Experiment_2_Error"] = (
    selected["Actual"] - selected["ANN_Experiment_2"]
).abs()

selected["Error_Improvement"] = (
    selected["Original_ANN_Error"]
    - selected["Experiment_2_Error"]
)

selected.to_csv(
    REPORT_FOLDER / "aircraft_experiment2_four_month_analysis.csv",
    index=False
)

# 18. Print results

print("\n========== MODEL COMPARISON ==========")

print(
    comparison.round(4).to_string(index=False)
)

print("\n========== FOUR-MONTH ERROR ANALYSIS ==========")

print(
    selected[
        [
            "YearMonth",
            "Actual",
            "Original_ANN",
            "ANN_Experiment_2",
            "Original_ANN_Error",
            "Experiment_2_Error",
            "Error_Improvement"
        ]
    ].round(2).to_string(index=False)
)

print("\n========== TRAINING SUMMARY ==========")

print("ANN Epochs Completed:", len(history.history["loss"]))

print(
    "Best Validation Loss:",
    round(min(history.history["val_loss"]), 6)
)

print("\nModels saved in:")
print(MODEL_FOLDER)

print("\nReports saved in:")
print(REPORT_FOLDER)

print("\n========== EXPERIMENT 2 COMPLETED ==========")
