
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path.cwd()

DATA_PATH = ROOT / "data/features/cargo_advanced_features.csv"
REPORT_DIR = ROOT / "reports/cargo_feature_analysis"

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

X = train[FEATURES].astype(float)
y = train[TARGET].astype(float)

print("CARGO FEATURE ANALYSIS")
print("=" * 60)
print("Training records:", len(train))
print("Number of features:", len(FEATURES))

# 1. Correlation analysis
correlation = train[FEATURES + [TARGET]].corr()

target_corr = correlation[TARGET].drop(TARGET)
target_corr = target_corr.reindex(
    target_corr.abs().sort_values(ascending=False).index
)

target_corr.to_csv(
    REPORT_DIR / "cargo_feature_target_correlation.csv"
)

print("\nFEATURE CORRELATION WITH TARGET")
print("=" * 60)
print(target_corr.to_string())

# 2. Feature-to-feature correlation
feature_corr = X.corr()

feature_corr.to_csv(
    REPORT_DIR / "cargo_feature_correlation_matrix.csv"
)

high_corr_pairs = []

for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        value = feature_corr.iloc[i, j]

        if abs(value) >= 0.90:
            high_corr_pairs.append({
                "Feature_1": FEATURES[i],
                "Feature_2": FEATURES[j],
                "Correlation": value
            })

high_corr_df = pd.DataFrame(high_corr_pairs)
high_corr_df.to_csv(
    REPORT_DIR / "cargo_high_correlation_pairs.csv",
    index=False
)

print("\nHIGHLY CORRELATED FEATURE PAIRS")
print("=" * 60)

if len(high_corr_df):
    print(high_corr_df.to_string(index=False))
else:
    print("No feature pairs above 0.90 correlation.")

# 3. Random Forest feature importance
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

rf_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

rf_importance.to_csv(
    REPORT_DIR / "cargo_random_forest_feature_importance.csv",
    index=False
)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)
print(rf_importance.to_string(index=False))

# 4. Extra Trees feature importance
et = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

et.fit(X, y)

et_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": et.feature_importances_
}).sort_values("Importance", ascending=False)

et_importance.to_csv(
    REPORT_DIR / "cargo_extra_trees_feature_importance.csv",
    index=False
)

# 5. Walk-forward permutation importance
tscv = TimeSeriesSplit(n_splits=4, test_size=24)

permutation_results = []

for fold, (fit_idx, val_idx) in enumerate(tscv.split(X), start=1):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    X_fit = X.iloc[fit_idx]
    X_val = X.iloc[val_idx]

    y_fit = y.iloc[fit_idx]
    y_val = y.iloc[val_idx]

    model.fit(X_fit, y_fit)

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    for feature, importance in zip(
        FEATURES,
        result.importances_mean
    ):
        permutation_results.append({
            "Fold": fold,
            "Feature": feature,
            "Importance": importance
        })

permutation_df = pd.DataFrame(permutation_results)

permutation_summary = (
    permutation_df.groupby("Feature")["Importance"]
    .agg(["mean", "std"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

permutation_summary.columns = [
    "Feature",
    "Mean_Permutation_Importance",
    "Std_Permutation_Importance"
]

permutation_summary.to_csv(
    REPORT_DIR / "cargo_walk_forward_permutation_importance.csv",
    index=False
)

print("\nWALK-FORWARD PERMUTATION IMPORTANCE")
print("=" * 60)
print(permutation_summary.to_string(index=False))

# 6. Visualizations
plt.figure(figsize=(12, 9))
sns.barplot(
    data=rf_importance.head(15),
    x="Importance",
    y="Feature"
)
plt.title("Top 15 Cargo Features - Random Forest")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "cargo_random_forest_top_features.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(14, 11))
sns.heatmap(
    feature_corr,
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Cargo Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "cargo_feature_correlation_heatmap.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(12, 9))
sns.barplot(
    data=permutation_summary.head(15),
    x="Mean_Permutation_Importance",
    y="Feature"
)
plt.title("Top 15 Cargo Features - Walk-Forward Permutation")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "cargo_permutation_importance.png",
    dpi=300
)
plt.close()

print("\nFEATURE ANALYSIS COMPLETED")
print("Reports saved in:", REPORT_DIR)
