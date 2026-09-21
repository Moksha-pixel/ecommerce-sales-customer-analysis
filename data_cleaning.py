import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# YUVAINTERN WEEK 2
# DATA ACQUISITION AND CLEANING PROCESS
# ============================================================

RAW_FILE = "online retail.xlsx"
CLEANED_FILE = "online_retail_cleaned.csv"
MODEL_FILE = "online_retail_model_ready.csv"
RESULT_FILE = "week2_results.txt"

# ------------------------------------------------------------
# 1. DATA ACQUISITION
# ------------------------------------------------------------

print("=" * 70)
print("YUVAINTERN WEEK 2 - DATA ACQUISITION AND CLEANING")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_excel(RAW_FILE)

original_rows = len(df)
original_columns = len(df.columns)

print("Dataset loaded successfully.")
print("Original rows:", original_rows)
print("Original columns:", original_columns)

# ------------------------------------------------------------
# 2. INITIAL PROFILING
# ------------------------------------------------------------

missing_before = df.isnull().sum()
duplicate_rows = df.duplicated().sum()

cancelled_mask = df["InvoiceNo"].astype(str).str.startswith("C")
negative_quantity_mask = df["Quantity"] < 0
invalid_price_mask = df["UnitPrice"] <= 0

negative_quantity_count = negative_quantity_mask.sum()
cancelled_count = cancelled_mask.sum()
invalid_price_count = invalid_price_mask.sum()

# ------------------------------------------------------------
# 3. SAVE INVESTIGATION EXAMPLES
# ------------------------------------------------------------

negative_examples = df[negative_quantity_mask].head(10)
cancelled_examples = df[cancelled_mask].head(10)
invalid_price_examples = df[invalid_price_mask].head(10)

# ------------------------------------------------------------
# 4. CLEANING
# ------------------------------------------------------------

clean = df.copy()

# A. Remove exact duplicate rows
clean = clean.drop_duplicates()

rows_after_duplicates = len(clean)

# B. Remove cancelled invoices
clean = clean[
    ~clean["InvoiceNo"].astype(str).str.startswith("C")
].copy()

rows_after_cancellations = len(clean)

# C. Remove records with non-positive quantity
# These records are not suitable for the normal sales dataset.
clean = clean[clean["Quantity"] > 0].copy()

rows_after_quantity = len(clean)

# D. Remove records with non-positive unit price
clean = clean[clean["UnitPrice"] > 0].copy()

rows_after_price = len(clean)

# E. Handle missing descriptions
clean["Description"] = clean["Description"].fillna("Unknown Product")

# F. Standardize text fields
clean["Description"] = clean["Description"].astype(str).str.strip()
clean["Country"] = clean["Country"].astype(str).str.strip()
clean["StockCode"] = clean["StockCode"].astype(str).str.strip()
clean["InvoiceNo"] = clean["InvoiceNo"].astype(str).str.strip()

# G. Ensure correct data types
clean["Quantity"] = pd.to_numeric(clean["Quantity"], errors="coerce")
clean["UnitPrice"] = pd.to_numeric(clean["UnitPrice"], errors="coerce")
clean["CustomerID"] = pd.to_numeric(clean["CustomerID"], errors="coerce")
clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"], errors="coerce")

# H. Remove rows where essential numerical/date fields became invalid
clean = clean.dropna(
    subset=["Quantity", "UnitPrice", "InvoiceDate"]
).copy()

# ------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ------------------------------------------------------------

# Transaction revenue
clean["Revenue"] = clean["Quantity"] * clean["UnitPrice"]

# Time-based features
clean["Year"] = clean["InvoiceDate"].dt.year
clean["Month"] = clean["InvoiceDate"].dt.month
clean["Month_Name"] = clean["InvoiceDate"].dt.strftime("%B")
clean["Day"] = clean["InvoiceDate"].dt.day
clean["Day_of_Week"] = clean["InvoiceDate"].dt.day_name()

# ------------------------------------------------------------
# 6. OUTLIER DETECTION
# ------------------------------------------------------------

# IQR method for Quantity and UnitPrice
def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


quantity_lower, quantity_upper = iqr_bounds(clean["Quantity"])
price_lower, price_upper = iqr_bounds(clean["UnitPrice"])
revenue_lower, revenue_upper = iqr_bounds(clean["Revenue"])

clean["Quantity_Outlier"] = (
    (clean["Quantity"] < quantity_lower)
    | (clean["Quantity"] > quantity_upper)
)

clean["UnitPrice_Outlier"] = (
    (clean["UnitPrice"] < price_lower)
    | (clean["UnitPrice"] > price_upper)
)

clean["Revenue_Outlier"] = (
    (clean["Revenue"] < revenue_lower)
    | (clean["Revenue"] > revenue_upper)
)

clean["Is_Outlier"] = (
    clean["Quantity_Outlier"]
    | clean["UnitPrice_Outlier"]
    | clean["Revenue_Outlier"]
)

outlier_count = clean["Is_Outlier"].sum()

# ------------------------------------------------------------
# 7. SAVE CLEANED DATASET
# ------------------------------------------------------------

clean.to_csv(CLEANED_FILE, index=False)

# ------------------------------------------------------------
# 8. CREATE MODEL-READY DATA
# ------------------------------------------------------------

# Use a copy without individual identifier fields that are not
# useful as numerical model features.
model_df = clean.copy()

# Keep only complete customer records for customer-level modeling.
# Transaction-level sales analysis can still use the main cleaned file.
model_df = model_df.dropna(subset=["CustomerID"]).copy()

# Remove outliers from this modeling dataset.
model_df = model_df[~model_df["Is_Outlier"]].copy()

# Select numerical variables for normalization.
numeric_features = [
    "Quantity",
    "UnitPrice",
    "Revenue"
]

scaler = MinMaxScaler()

model_df[numeric_features] = scaler.fit_transform(
    model_df[numeric_features]
)

# One-hot encode Country.
model_df = pd.get_dummies(
    model_df,
    columns=["Country"],
    prefix="Country",
    dtype=int
)

# Remove helper outlier columns after filtering.
model_df = model_df.drop(
    columns=[
        "Quantity_Outlier",
        "UnitPrice_Outlier",
        "Revenue_Outlier",
        "Is_Outlier"
    ],
    errors="ignore"
)

model_df.to_csv(MODEL_FILE, index=False)

# ------------------------------------------------------------
# 9. FINAL VALIDATION
# ------------------------------------------------------------

missing_after = clean.isnull().sum()
duplicate_after = clean.duplicated().sum()

final_rows = len(clean)
final_columns = len(clean.columns)

# ------------------------------------------------------------
# 10. CREATE WEEK 2 RESULTS REPORT
# ------------------------------------------------------------

with open(RESULT_FILE, "w", encoding="utf-8") as f:

    f.write("YUVAINTERN WEEK 2 - DATA ACQUISITION AND CLEANING RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. DATA ACQUISITION\n")
    f.write("-" * 40 + "\n")
    f.write("Source: UCI Machine Learning Repository - Online Retail Dataset\n")
    f.write(f"Input file: {RAW_FILE}\n")
    f.write(f"Original rows: {original_rows}\n")
    f.write(f"Original columns: {original_columns}\n\n")

    f.write("2. INITIAL DATA QUALITY PROFILE\n")
    f.write("-" * 40 + "\n")
    f.write(f"Duplicate rows: {duplicate_rows}\n")
    f.write(f"Cancelled invoice records: {cancelled_count}\n")
    f.write(f"Negative quantity records: {negative_quantity_count}\n")
    f.write(f"Zero/negative price records: {invalid_price_count}\n\n")

    f.write("Missing values before cleaning:\n")
    f.write(missing_before.to_string())
    f.write("\n\n")

    f.write("3. CLEANING PROCESS\n")
    f.write("-" * 40 + "\n")
    f.write(
        f"Rows after removing exact duplicate rows: "
        f"{rows_after_duplicates}\n"
    )
    f.write(
        f"Rows after removing cancelled invoices: "
        f"{rows_after_cancellations}\n"
    )
    f.write(
        f"Rows after removing non-positive quantities: "
        f"{rows_after_quantity}\n"
    )
    f.write(
        f"Rows after removing non-positive prices: "
        f"{rows_after_price}\n"
    )
    f.write(
        "Missing descriptions were replaced with 'Unknown Product'.\n"
    )
    f.write(
        "Text fields were stripped of unnecessary leading/trailing spaces.\n"
    )
    f.write(
        "InvoiceDate was converted to datetime format.\n"
    )
    f.write(
        "Quantity, UnitPrice and CustomerID were converted to numeric types.\n"
    )
    f.write(
        "Rows with invalid Quantity, UnitPrice or InvoiceDate were removed.\n\n"
    )

    f.write("4. FEATURE ENGINEERING\n")
    f.write("-" * 40 + "\n")
    f.write("Revenue = Quantity × UnitPrice\n")
    f.write("Created Year, Month, Month_Name, Day and Day_of_Week.\n\n")

    f.write("5. OUTLIER DETECTION\n")
    f.write("-" * 40 + "\n")
    f.write("Method: Interquartile Range (IQR)\n")
    f.write(f"Quantity lower bound: {quantity_lower:.4f}\n")
    f.write(f"Quantity upper bound: {quantity_upper:.4f}\n")
    f.write(f"UnitPrice lower bound: {price_lower:.4f}\n")
    f.write(f"UnitPrice upper bound: {price_upper:.4f}\n")
    f.write(f"Revenue lower bound: {revenue_lower:.4f}\n")
    f.write(f"Revenue upper bound: {revenue_upper:.4f}\n")
    f.write(f"Outlier records identified: {outlier_count}\n")
    f.write(
        "Outliers were flagged rather than automatically deleted from "
        "the main cleaned transaction dataset because extreme values may "
        "represent genuine wholesale transactions.\n\n"
    )

    f.write("6. NORMALIZATION AND ENCODING\n")
    f.write("-" * 40 + "\n")
    f.write(
        "Quantity, UnitPrice and Revenue were normalized using Min-Max scaling "
        "for the model-ready dataset.\n"
    )
    f.write(
        "Country was converted into numerical features using one-hot encoding.\n"
    )
    f.write(
        "CustomerID was retained as an identifier for customer analysis and "
        "not treated as a continuous numerical feature.\n\n"
    )

    f.write("7. FINAL VALIDATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Final cleaned rows: {final_rows}\n")
    f.write(f"Final cleaned columns: {final_columns}\n")
    f.write(f"Duplicate rows after cleaning: {duplicate_after}\n\n")

    f.write("Missing values after cleaning:\n")
    f.write(missing_after.to_string())
    f.write("\n\n")

    f.write("8. OUTPUT FILES\n")
    f.write("-" * 40 + "\n")
    f.write(f"Cleaned dataset: {CLEANED_FILE}\n")
    f.write(f"Model-ready dataset: {MODEL_FILE}\n")
    f.write(f"Results report: {RESULT_FILE}\n")

# ------------------------------------------------------------
# 11. DISPLAY FINAL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLEANING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOriginal rows:", original_rows)
print("Final cleaned rows:", final_rows)
print("Rows removed:", original_rows - final_rows)

print("\nDuplicate rows before:", duplicate_rows)
print("Duplicate rows after:", duplicate_after)

print("\nOutliers identified:", outlier_count)

print("\nFiles created:")
print("1.", CLEANED_FILE)
print("2.", MODEL_FILE)
print("3.", RESULT_FILE)

print("\n" + "=" * 70)
print("WEEK 2 DATA PROCESSING COMPLETE")
print("=" * 70)