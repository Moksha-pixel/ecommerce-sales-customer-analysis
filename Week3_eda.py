import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# YUVAINTERN WEEK 3: EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# Project: E-Commerce Sales and Customer Analytics
# ============================================================

INPUT_FILE = "online_retail_cleaned.csv"
OUTPUT_FOLDER = "Week3_Results"
CHART_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")

os.makedirs(CHART_FOLDER, exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 65)
print("YUVAINTERN WEEK 3 - EDA AND VISUALIZATION")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n1. LOADING CLEANED DATASET")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}. "
        "Save week3_eda.py in the same folder as the cleaned CSV."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country", "Revenue"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Required columns missing: {missing_columns}")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice", "Revenue"])

# Use a copy of the loaded data; the original CSV is not changed.
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

# ------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS AND DATA SUMMARY
# ------------------------------------------------------------

print("\n2. DESCRIPTIVE STATISTICS")

summary = df[["Quantity", "UnitPrice", "Revenue"]].describe().round(2)
print(summary.to_string())

country_count = df["Country"].nunique()
product_count = df["StockCode"].nunique()
invoice_count = df["InvoiceNo"].nunique()
customer_count = df["CustomerID"].nunique()

total_revenue = df["Revenue"].sum()
average_order_line_revenue = df["Revenue"].mean()
median_line_revenue = df["Revenue"].median()

missing_values = df[
    ["InvoiceNo", "StockCode", "Description", "Quantity",
     "InvoiceDate", "UnitPrice", "CustomerID", "Country", "Revenue"]
].isnull().sum()

# ------------------------------------------------------------
# 3. MONTHLY REVENUE TREND
# ------------------------------------------------------------

monthly_revenue = (
    df.groupby("YearMonth")["Revenue"]
    .sum()
    .sort_index()
)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_revenue.index,
    monthly_revenue.values,
    marker="o",
    linewidth=2
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "01_monthly_revenue.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 4. TOP 10 COUNTRIES BY REVENUE
# ------------------------------------------------------------

country_revenue = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

top_countries = country_revenue.head(10).sort_values()

plt.figure(figsize=(10, 6))
top_countries.plot(kind="barh")
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "02_top_countries.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 5. REVENUE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.histplot(df["Revenue"], bins=50, kde=True)
plt.xlim(0, df["Revenue"].quantile(0.99))
plt.title("Revenue Distribution (Up to 99th Percentile)")
plt.xlabel("Revenue per Transaction Line")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "03_revenue_distribution.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 6. QUANTITY DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.histplot(df["Quantity"], bins=50, kde=False)
plt.xlim(0, df["Quantity"].quantile(0.99))
plt.title("Quantity Distribution (Up to 99th Percentile)")
plt.xlabel("Quantity per Transaction Line")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "04_quantity_distribution.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 7. BOXPLOT: UNIT PRICE
# ------------------------------------------------------------

price_limit = df["UnitPrice"].quantile(0.99)
price_for_plot = df.loc[df["UnitPrice"] <= price_limit, "UnitPrice"]

plt.figure(figsize=(10, 4))
sns.boxplot(x=price_for_plot)
plt.title("Unit Price Box Plot (Values up to 99th Percentile)")
plt.xlabel("Unit Price")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "05_unit_price_boxplot.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 8. SCATTER PLOT: QUANTITY VS REVENUE
# ------------------------------------------------------------

scatter_data = df.sample(n=min(5000, len(df)), random_state=42)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=scatter_data,
    x="Quantity",
    y="Revenue",
    alpha=0.4,
    s=20
)
plt.xlim(0, df["Quantity"].quantile(0.99))
plt.ylim(0, df["Revenue"].quantile(0.99))
plt.title("Quantity vs Revenue (Sample of Up to 5,000 Rows)")
plt.xlabel("Quantity")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "06_quantity_vs_revenue.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 9. CORRELATION HEATMAP
# ------------------------------------------------------------

correlation = df[["Quantity", "UnitPrice", "Revenue"]].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "07_correlation_heatmap.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 10. TOP 10 PRODUCTS BY REVENUE
# ------------------------------------------------------------

product_revenue = (
    df.groupby(["StockCode", "Description"], dropna=False)["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

product_labels = [
    f"{stock} - {str(description)[:35]}"
    for stock, description in product_revenue.index
]

plt.figure(figsize=(12, 7))
plt.barh(product_labels[::-1], product_revenue.values[::-1])
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "08_top_products.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 11. CUSTOMER REVENUE ANALYSIS
# ------------------------------------------------------------

identified_customers = df.dropna(subset=["CustomerID"])
customer_revenue = identified_customers.groupby("CustomerID")["Revenue"].sum()

top_customers = customer_revenue.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_customers.sort_values().plot(kind="barh")
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Customer ID")
plt.tight_layout()
plt.savefig(os.path.join(CHART_FOLDER, "09_top_customers.png"), dpi=200)
plt.close()

# ------------------------------------------------------------
# 12. AUTOMATIC FINDINGS AND REPORT FILE
# ------------------------------------------------------------

best_month = monthly_revenue.idxmax()
best_month_revenue = monthly_revenue.max()

best_country = country_revenue.idxmax()
best_country_revenue = country_revenue.max()

best_product = product_revenue.index[0]
best_product_revenue = product_revenue.iloc[0]

quantity_revenue_corr = df["Quantity"].corr(df["Revenue"])
price_revenue_corr = df["UnitPrice"].corr(df["Revenue"])

results_path = os.path.join(OUTPUT_FOLDER, "week3_eda_results.txt")

with open(results_path, "w", encoding="utf-8") as file:
    file.write("YUVAINTERN WEEK 3: EDA RESULTS\n")
    file.write("=" * 55 + "\n\n")

    file.write("1. DATA SUMMARY\n")
    file.write(f"Rows analysed: {len(df):,}\n")
    file.write(f"Countries: {country_count}\n")
    file.write(f"Unique products: {product_count:,}\n")
    file.write(f"Unique invoices: {invoice_count:,}\n")
    file.write(f"Unique identified customers: {customer_count:,}\n")
    file.write(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}\n")
    file.write(f"Total revenue across analysed transaction lines: {total_revenue:,.2f}\n")
    file.write(f"Average line revenue: {average_order_line_revenue:,.2f}\n")
    file.write(f"Median line revenue: {median_line_revenue:,.2f}\n\n")

    file.write("2. DESCRIPTIVE STATISTICS\n")
    file.write(summary.to_string())
    file.write("\n\n3. MISSING VALUES\n")
    file.write(missing_values.to_string())
    file.write("\n\n4. VISUALIZATION FINDINGS\n")
    file.write(
        f"Highest-revenue month: {best_month} "
        f"({best_month_revenue:,.2f})\n"
    )
    file.write(
        f"Highest-revenue country: {best_country} "
        f"({best_country_revenue:,.2f})\n"
    )
    file.write(
        f"Top product by revenue: {best_product} "
        f"({best_product_revenue:,.2f})\n"
    )
    file.write(
        f"Correlation between Quantity and Revenue: "
        f"{quantity_revenue_corr:.3f}\n"
    )
    file.write(
        f"Correlation between UnitPrice and Revenue: "
        f"{price_revenue_corr:.3f}\n\n"
    )

    file.write("5. INTERPRETATION NOTES\n")
    file.write(
        "- Monthly revenue shows how recorded sales value changes over time.\n"
    )
    file.write(
        "- Country and product charts identify the largest revenue contributors.\n"
    )
    file.write(
        "- Histograms show the distributions and skewness of revenue and quantity.\n"
    )
    file.write(
        "- The box plot helps display the spread and extreme unit-price values.\n"
    )
    file.write(
        "- The scatter plot explores the relationship between quantity and revenue.\n"
    )
    file.write(
        "- Correlation measures linear association, not causation.\n"
    )
    file.write(
        "- The cleaned dataset excludes rows removed during Week 2; "
        "results describe this prepared dataset, not every original transaction.\n"
    )
    file.write(
        "- Charts that limit the displayed range are zoomed for readability; "
        "the underlying dataset is not trimmed for those charts.\n"
    )

# ------------------------------------------------------------
# 13. FINISH
# ------------------------------------------------------------

print("\n3. KEY FINDINGS")
print(f"Total revenue: {total_revenue:,.2f}")
print(f"Highest-revenue month: {best_month} ({best_month_revenue:,.2f})")
print(f"Highest-revenue country: {best_country} ({best_country_revenue:,.2f})")
print(f"Top product by revenue: {best_product} ({best_product_revenue:,.2f})")
print(f"Quantity-Revenue correlation: {quantity_revenue_corr:.3f}")

print("\n4. FILES CREATED")
print(f"Results: {results_path}")
print(f"Charts folder: {CHART_FOLDER}")

print("\nCharts created:")
for filename in sorted(os.listdir(CHART_FOLDER)):
    if filename.endswith(".png"):
        print(f" - {filename}")

print("\n" + "=" * 65)
print("WEEK 3 EDA COMPLETED SUCCESSFULLY")
print("=" * 65)