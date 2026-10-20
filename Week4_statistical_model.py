import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. File paths and configuration
# --------------------------------------------------
INPUT_FILE = "online_retail_cleaned.csv"
OUTPUT_DIR = "Week4_Results"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(CHART_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. Run this script from your project folder."
    )

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = [
    "Quantity", "UnitPrice", "InvoiceDate", "Country"
]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Required columns missing: {missing_columns}")

# --------------------------------------------------
# 2. Prepare data for quantity prediction
# --------------------------------------------------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

df = df.dropna(
    subset=["InvoiceDate", "Quantity", "UnitPrice", "Country"]
).copy()

# Remove invalid or non-positive values for this modeling task.
# This is an additional modeling filter, not a change to the source CSV.
df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

df["Month"] = df["InvoiceDate"].dt.month
df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
df["Hour"] = df["InvoiceDate"].dt.hour

# Use a representative sample to reduce training time
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)

print("Rows used after sampling:", len(df))

features = ["UnitPrice", "Month", "DayOfWeek", "Hour", "Country"]
target = "Quantity"

X = df[features]
y = df[target]

if len(df) < 100:
    raise ValueError("Too few valid rows for reliable model evaluation.")

print(f"Rows used for modeling: {len(df):,}")
print(f"Features: {features}")
print(f"Target: {target}")

# --------------------------------------------------
# 3. Split training and testing data
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")

# --------------------------------------------------
# 4. Build preprocessing and model pipeline
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("country", OneHotEncoder(handle_unknown="ignore"),
         ["Country"]),
        ("numeric", "passthrough",
         ["UnitPrice", "Month", "DayOfWeek", "Hour"])
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# --------------------------------------------------
# 5. Parameter tuning with cross-validation
# --------------------------------------------------
parameter_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1, 4]
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=6,
    scoring="neg_mean_absolute_error",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=1
)

print("\nTraining model and tuning parameters...")
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Model training and tuning completed.")

# --------------------------------------------------
# 6. Evaluate on held-out test data
# --------------------------------------------------
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Baseline: always predict the training-set median.
baseline_predictions = np.full(len(y_test), y_train.median())
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("\nTEST RESULTS")
print("-" * 45)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Baseline median-prediction MAE: {baseline_mae:.4f}")
print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation MAE: {-search.best_score_:.4f}")

# --------------------------------------------------
# 7. Save results and sample predictions
# --------------------------------------------------
results_path = os.path.join(OUTPUT_DIR, "week4_model_results.txt")
predictions_path = os.path.join(OUTPUT_DIR, "week4_test_predictions.csv")

with open(results_path, "w", encoding="utf-8") as f:
    f.write("YUVAINTERN WEEK 4: STATISTICAL MODELING RESULTS\n")
    f.write("=" * 55 + "\n\n")
    f.write("MODEL: Random Forest Regression\n")
    f.write("Target: Quantity\n")
    f.write("Features: UnitPrice, Month, DayOfWeek, Hour, Country\n\n")
    f.write("DATA PREPARATION\n")
    f.write(f"Rows used after modeling filters: {len(df)}\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Testing rows: {len(X_test)}\n")
    f.write("Test split: 20%; random_state=42\n")
    f.write("Cross-validation: 3-fold shuffled KFold\n\n")
    f.write("TEST METRICS\n")
    f.write(f"MAE: {mae:.6f}\n")
    f.write(f"RMSE: {rmse:.6f}\n")
    f.write(f"R-squared: {r2:.6f}\n")
    f.write(f"Baseline median-prediction MAE: {baseline_mae:.6f}\n")
    f.write(f"Best CV MAE: {-search.best_score_:.6f}\n\n")
    f.write(f"Best parameters: {search.best_params_}\n\n")
    f.write("LIMITATIONS\n")
    f.write(
        "The random split may place transactions from the same invoice "
        "or time period in both training and test sets. Results therefore "
        "measure performance on a random holdout, not future-period "
        "forecasting. Quantity can also be highly skewed.\n"
    )

prediction_df = pd.DataFrame({
    "Actual_Quantity": y_test.to_numpy(),
    "Predicted_Quantity": predictions
})
prediction_df.to_csv(predictions_path, index=False)

# --------------------------------------------------
# 8. Create evaluation charts
# --------------------------------------------------
plt.figure(figsize=(8, 6))
sample_size = min(3000, len(y_test))
sample_indices = np.random.default_rng(42).choice(
    len(y_test), size=sample_size, replace=False
)
plt.scatter(
    y_test.iloc[sample_indices],
    predictions[sample_indices],
    alpha=0.35
)
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")
plt.title("Actual vs Predicted Quantity (Test Sample)")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "01_actual_vs_predicted.png"),
    dpi=150
)
plt.close()

errors = y_test.to_numpy() - predictions
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Test Prediction Errors")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "02_prediction_errors.png"),
    dpi=150
)
plt.close()

metric_names = ["MAE", "RMSE", "R-squared"]
metric_values = [mae, rmse, r2]

plt.figure(figsize=(8, 5))
plt.bar(metric_names, metric_values)
plt.title("Random Forest Test Metrics")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig(
    os.path.join(CHART_DIR, "03_model_metrics.png"),
    dpi=150
)
plt.close()

print("\nFiles saved:")
print(results_path)
print(predictions_path)
print(CHART_DIR)
print("\nWeek 4 modeling script finished.")