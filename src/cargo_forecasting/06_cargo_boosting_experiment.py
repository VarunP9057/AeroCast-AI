
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/cargo_advanced_features.csv"
REPORT_DIR = ROOT / "reports/cargo_forecasting_experiment4"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Cargo Weight LBS"

FEATURES = [
    "Year", "Month", "Quarter",
    "Month_Sin", "Month_Cos", "Time_Index",
    "Lag_1", "Lag_2", "Lag_3",
    "Lag_6", "Lag_12", "Lag_18", "Lag_24",
    "Rolling_Mean_3", "Rolling_Mean_6",
    "Rolling_Mean_12", "Rolling_Mean_24",
    "Rolling_Std_3", "Rolling_Std_6", "Rolling_Std_12",
    "Previous_Year_Change", "Previous_Year_Change_Pct",
    "Trend_Slope_6", "Trend_Slope_12"
]

df = pd.read_csv(DATA_PATH)
df["YearMonth"] = pd.to_datetime(df["YearMonth"])
df = df.sort_values("YearMonth").reset_index(drop=True)

train = df[df["YearMonth"] < "2021-06-01"].copy()
test = df[df["YearMonth"] >= "2021-06-01"].copy()

X_train = train[FEATURES].astype(float)
y_train = train[TARGET].astype(float)

X_test = test[FEATURES].astype(float)
y_test = test[TARGET].astype(float)

print("CARGO EXPERIMENT 4 - BOOSTING MODELS")
print("=" * 60)
print("Training records:", len(train))
print("Untouched test records:", len(test))
print("Features:", len(FEATURES))

MODELS = {
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        depth=4,
        learning_rate=0.03,
        loss_function="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        thread_count=-1
    ),

    "Gradient_Boosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        loss="huber",
        random_state=42
    ),

    "Extra_Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Random_Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
}

tscv = TimeSeriesSplit(n_splits=4, test_size=24)

results = []

for model_name, model in MODELS.items():

    print(f"\nEvaluating: {model_name}")
    print("-" * 50)

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for fold, (fit_idx, val_idx) in enumerate(
        tscv.split(X_train), start=1
    ):

        X_fit = X_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]

        y_fit = y_train.iloc[fit_idx]
        y_val = y_train.iloc[val_idx]

        model.fit(X_fit, y_fit)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        print(
            f"Fold {fold}: "
            f"MAE={mae:,.0f}, "
            f"RMSE={rmse:,.0f}, "
            f"R2={r2:.4f}"
        )

    results.append({
        "Model": model_name,
        "Average_MAE": np.mean(fold_mae),
        "Average_RMSE": np.mean(fold_rmse),
        "Average_R2": np.mean(fold_r2)
    })

    print(f"Average MAE: {np.mean(fold_mae):,.0f}")


results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Average_MAE")

results_df.to_csv(
    REPORT_DIR / "cargo_experiment4_validation_results.csv",
    index=False
)

print("\nFINAL WALK-FORWARD MODEL COMPARISON")
print("=" * 75)
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]["Model"]

print("\nBest model:", best_model)
print("Selection criterion: Lowest average validation MAE")
print("\nThe 63-month test dataset has NOT been used.")
