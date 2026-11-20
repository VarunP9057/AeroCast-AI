
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "features",
    "aircraft_monthly_features.csv"
)

MODEL_DIR = os.path.join(
    ROOT_DIR,
    "models",
    "aircraft_forecasting_experiment3"
)

REPORT_DIR = os.path.join(
    ROOT_DIR,
    "reports",
    "aircraft_forecasting_experiment3"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


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
    "Rolling_Std_3",
    "Rolling_Std_6"
]

TARGET = "Landing Count"

CONFIGURATIONS = [
    {
        "name": "Small_ANN",
        "layers": [32, 16],
        "dropout": 0.10,
        "l2": 0.001,
        "learning_rate": 0.001
    },
    {
        "name": "Regularized_ANN",
        "layers": [32, 16],
        "dropout": 0.20,
        "l2": 0.005,
        "learning_rate": 0.0005
    },
    {
        "name": "Balanced_ANN",
        "layers": [64, 32],
        "dropout": 0.10,
        "l2": 0.001,
        "learning_rate": 0.0005
    }
]


def build_model(config, input_features):

    model = Sequential()

    model.add(Input(shape=(input_features,)))

    for units in config["layers"]:

        model.add(
            Dense(
                units,
                activation="relu",
                kernel_regularizer=l2(config["l2"])
            )
        )

        model.add(Dropout(config["dropout"]))

    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"]
    )

    model.compile(
        optimizer=optimizer,
        loss="mse"
    )

    return model


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


def train_model(model, X_train, y_train, X_val, y_val):

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

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=[early_stopping, reduce_lr],
        shuffle=False,
        verbose=0
    )

    return history


print("\nAIRCRAFT FORECASTING - EXPERIMENT 3")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["YearMonth"] = pd.to_datetime(
    df["YearMonth"].astype(str),
    format="%Y-%m"
)

df = df.sort_values("YearMonth").reset_index(drop=True)

print("\nDataset shape:", df.shape)
print("Date range:", df["YearMonth"].min(), "to", df["YearMonth"].max())

missing = df[FEATURES + [TARGET]].isnull().sum().sum()

if missing > 0:
    raise ValueError(
        f"Dataset contains {missing} missing feature or target values."
    )

X = df[FEATURES].values.astype("float32")
y = df[TARGET].values.astype("float32")

test_start = pd.Timestamp("2021-06-01")

train_mask = df["YearMonth"] < test_start
test_mask = df["YearMonth"] >= test_start

X_train_all = X[train_mask]
y_train_all = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

test_dates = df.loc[test_mask, "YearMonth"].reset_index(drop=True)

print("\nTraining records:", len(X_train_all))
print("Testing records:", len(X_test))
print("Features:", len(FEATURES))

if len(X_test) != 63:
    raise ValueError(
        "Expected 63 test records. Check the dataset date range."
    )


print("\nSTEP 1: WALK-FORWARD VALIDATION")
print("=" * 60)

time_split = TimeSeriesSplit(
    n_splits=4,
    test_size=24
)

validation_results = []

for config_index, config in enumerate(CONFIGURATIONS):

    print(f"\nTesting configuration: {config['name']}")

    fold_scores = []

    for fold, (train_indices, val_indices) in enumerate(
        time_split.split(X_train_all), start=1
    ):

        X_fold_train = X_train_all[train_indices]
        y_fold_train = y_train_all[train_indices]

        X_fold_val = X_train_all[val_indices]
        y_fold_val = y_train_all[val_indices]

        inner_split = int(len(X_fold_train) * 0.80)

        X_fit = X_fold_train[:inner_split]
        y_fit = y_fold_train[:inner_split]

        X_inner_val = X_fold_train[inner_split:]
        y_inner_val = y_fold_train[inner_split:]

        scaler = StandardScaler()
        target_scaler = StandardScaler()

        X_fit_scaled = scaler.fit_transform(X_fit)
        X_inner_val_scaled = scaler.transform(X_inner_val)
        X_fold_val_scaled = scaler.transform(X_fold_val)

        y_fit_scaled = target_scaler.fit_transform(
            y_fit.reshape(-1, 1)
        )

        y_inner_val_scaled = target_scaler.transform(
            y_inner_val.reshape(-1, 1)
        )

        tf.keras.backend.clear_session()

        tf.keras.utils.set_random_seed(
            RANDOM_STATE + config_index * 10 + fold
        )

        model = build_model(config, len(FEATURES))

        train_model(
            model,
            X_fit_scaled,
            y_fit_scaled,
            X_inner_val_scaled,
            y_inner_val_scaled
        )

        predictions_scaled = model.predict(
            X_fold_val_scaled,
            verbose=0
        )

        predictions = target_scaler.inverse_transform(
            predictions_scaled
        ).flatten()

        metrics = calculate_metrics(
            y_fold_val,
            predictions
        )

        fold_scores.append(metrics["MAE"])

        print(
            f"Fold {fold} | "
            f"MAE: {metrics['MAE']:.2f} | "
            f"RMSE: {metrics['RMSE']:.2f} | "
            f"R2: {metrics['R2']:.4f}"
        )

    average_mae = np.mean(fold_scores)

    validation_results.append({
        "Model": config["name"],
        "Average_Validation_MAE": average_mae,
        "Fold_1_MAE": fold_scores[0],
        "Fold_2_MAE": fold_scores[1],
        "Fold_3_MAE": fold_scores[2],
        "Fold_4_MAE": fold_scores[3]
    })

    print(
        f"Average validation MAE: {average_mae:.2f}"
    )


validation_df = pd.DataFrame(validation_results)

validation_df = validation_df.sort_values(
    "Average_Validation_MAE"
).reset_index(drop=True)

validation_df.to_csv(
    os.path.join(REPORT_DIR, "walk_forward_validation.csv"),
    index=False
)

print("\nWALK-FORWARD VALIDATION RESULTS")
print(validation_df.to_string(index=False))


best_model_name = validation_df.iloc[0]["Model"]

best_config = next(
    config for config in CONFIGURATIONS
    if config["name"] == best_model_name
)

print("\nSelected configuration:", best_model_name)


print("\nSTEP 2: FINAL MODEL TRAINING")
print("=" * 60)

final_split = int(len(X_train_all) * 0.80)

X_final_fit = X_train_all[:final_split]
y_final_fit = y_train_all[:final_split]

X_final_val = X_train_all[final_split:]
y_final_val = y_train_all[final_split:]

final_scaler = StandardScaler()
final_target_scaler = StandardScaler()

X_final_fit_scaled = final_scaler.fit_transform(X_final_fit)
X_final_val_scaled = final_scaler.transform(X_final_val)
X_test_scaled = final_scaler.transform(X_test)

y_final_fit_scaled = final_target_scaler.fit_transform(
    y_final_fit.reshape(-1, 1)
)

y_final_val_scaled = final_target_scaler.transform(
    y_final_val.reshape(-1, 1)
)

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(RANDOM_STATE)

final_model = build_model(
    best_config,
    len(FEATURES)
)

final_history = train_model(
    final_model,
    X_final_fit_scaled,
    y_final_fit_scaled,
    X_final_val_scaled,
    y_final_val_scaled
)

test_predictions_scaled = final_model.predict(
    X_test_scaled,
    verbose=0
)

test_predictions = final_target_scaler.inverse_transform(
    test_predictions_scaled
).flatten()

test_predictions = np.maximum(test_predictions, 0)

final_metrics = calculate_metrics(
    y_test,
    test_predictions
)

print("\nFINAL EXPERIMENT 3 TEST RESULTS")
print("=" * 60)

for metric, value in final_metrics.items():
    print(f"{metric}: {value:.4f}")


print("\nSTEP 3: SAVING MODEL AND REPORTS")
print("=" * 60)

final_model.save(
    os.path.join(MODEL_DIR, "aircraft_ann_experiment3.keras")
)

joblib.dump(
    final_scaler,
    os.path.join(MODEL_DIR, "aircraft_ann_scaler.pkl")
)

joblib.dump(
    final_target_scaler,
    os.path.join(MODEL_DIR, "aircraft_ann_target_scaler.pkl")
)

joblib.dump(
    FEATURES,
    os.path.join(MODEL_DIR, "aircraft_feature_names.pkl")
)

predictions_df = pd.DataFrame({
    "YearMonth": test_dates,
    "Actual": y_test,
    "Experiment3_ANN": test_predictions,
    "Absolute_Error": np.abs(y_test - test_predictions)
})

predictions_df.to_csv(
    os.path.join(REPORT_DIR, "aircraft_experiment3_predictions.csv"),
    index=False
)

comparison_df = pd.DataFrame([
    {
        "Model": "Experiment3_ANN",
        **final_metrics
    }
])

comparison_df.to_csv(
    os.path.join(REPORT_DIR, "aircraft_experiment3_metrics.csv"),
    index=False
)

history_df = pd.DataFrame(final_history.history)

history_df.to_csv(
    os.path.join(REPORT_DIR, "aircraft_experiment3_training_history.csv"),
    index=False
)

print("\nSAVED FILES:")
print("Model:", MODEL_DIR)
print("Reports:", REPORT_DIR)

print("\nEXPERIMENT 3 COMPLETED SUCCESSFULLY")
