
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("E-Commerce Sales Analytics Dashboard")
st.caption("Interactive Dashboard | YuvaIntern Week 5")

DATA_FILE = Path(__file__).parent / "online_retail_cleaned.csv"


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["Quantity"] = pd.to_numeric(
        data["Quantity"], errors="coerce"
    )
    data["UnitPrice"] = pd.to_numeric(
        data["UnitPrice"], errors="coerce"
    )

    data = data.dropna(
        subset=["InvoiceDate", "Quantity", "UnitPrice", "Country"]
    )

    data = data[
        (data["Quantity"] > 0) &
        (data["UnitPrice"] > 0)
    ].copy()

    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    data["Month"] = data["InvoiceDate"].dt.to_period("M").astype(str)

    return data


if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Place online_retail_cleaned.csv "
        "in the same folder as week5_dashboard.py."
    )
    st.stop()

with st.spinner("Loading transaction data..."):
    df = load_data(str(DATA_FILE))

if df.empty:
    st.warning("No valid sales transactions were found.")
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

countries = sorted(df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select countries",
    countries,
    default=countries
)

months = sorted(df["Month"].unique().tolist())
selected_months = st.sidebar.multiselect(
    "Select months",
    months,
    default=months
)

filtered = df[
    df["Country"].isin(selected_countries) &
    df["Month"].isin(selected_months)
].copy()

st.sidebar.caption("Change filters to update the dashboard.")

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

# KPI cards
total_revenue = filtered["Revenue"].sum()
total_quantity = filtered["Quantity"].sum()
invoice_count = filtered["InvoiceNo"].nunique()
country_count = filtered["Country"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"{total_revenue:,.2f}")
c2.metric("Quantity Sold", f"{total_quantity:,.0f}")
c3.metric("Unique Invoices", f"{invoice_count:,}")
c4.metric("Countries", f"{country_count:,}")

st.divider()

# Monthly revenue chart
st.subheader("Monthly Revenue Trend")

monthly = (
    filtered.groupby("Month", as_index=False)["Revenue"]
    .sum()
    .sort_values("Month")
)

fig_monthly = px.line(
    monthly,
    x="Month",
    y="Revenue",
    markers=True,
    title="Revenue by Month",
    labels={"Month": "Month", "Revenue": "Revenue"}
)

fig_monthly.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_monthly, use_container_width=True)

# Two-column chart layout
left, right = st.columns(2)

with left:
    st.subheader("Top 10 Countries")

    country_sales = (
        filtered.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .nlargest(10, "Revenue")
        .sort_values("Revenue")
    )

    fig_country = px.bar(
        country_sales,
        x="Revenue",
        y="Country",
        orientation="h",
        title="Top Countries by Revenue",
        labels={"Revenue": "Revenue", "Country": "Country"}
    )

    st.plotly_chart(fig_country, use_container_width=True)

with right:
    st.subheader("Top 10 Products")

    product_column = (
        "Description"
        if "Description" in filtered.columns
        else "StockCode"
    )

    product_sales = (
        filtered.groupby(product_column, as_index=False)["Revenue"]
        .sum()
        .nlargest(10, "Revenue")
        .sort_values("Revenue")
    )

    fig_product = px.bar(
        product_sales,
        x="Revenue",
        y=product_column,
        orientation="h",
        title="Top Products by Revenue",
        labels={"Revenue": "Revenue", product_column: "Product"}
    )

    st.plotly_chart(fig_product, use_container_width=True)

# Revenue distribution
st.subheader("Revenue Distribution")

fig_distribution = px.histogram(
    filtered,
    x="Revenue",
    nbins=50,
    title="Distribution of Transaction-Line Revenue",
    labels={"Revenue": "Revenue per transaction line"}
)

st.plotly_chart(fig_distribution, use_container_width=True)

# Filtered data table
st.subheader("Transaction Details")
st.caption(
    f"Showing up to 100 rows from {len(filtered):,} filtered transactions."
)

display_columns = [
    col for col in [
        "InvoiceNo", "InvoiceDate", "Description",
        "Quantity", "UnitPrice", "Revenue", "Country"
    ]
    if col in filtered.columns
]

st.dataframe(
    filtered[display_columns].head(100),
    use_container_width=True,
    hide_index=True
)

csv_data = filtered[display_columns].to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered data as CSV",
    data=csv_data,
    file_name="filtered_ecommerce_transactions.csv",
    mime="text/csv"
)

st.caption("YuvaIntern Week 5 | E-Commerce Sales and Customer Analytics")
