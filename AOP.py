import streamlit as st
import pandas as pd

# Industry‐standard default growth targets (editable via sliders)
BENCHMARKS = {
    "annual_volume_growth": 0.055,    # 5.5% midpoint of 4.5–6.5%
    "annual_value_growth": 0.055,     # assume same as volume growth
}


def load_clean_sales_data(filepath: str) -> pd.DataFrame:
    """
    Loads and cleans the sales CSV with:
    - Latin-1 encoding
    - Comma-stripped Redistribution Value → float
    - Day-first Delivered_date → datetime
    - Month period column
    - Fill Delivered Qty NaNs with zero
    - Compute Total_Amount_Spent
    - Auto-generate Order_Id if missing
    """
    df = pd.read_csv(filepath, encoding="latin1")
    df["Redistribution Value"] = (
        df["Redistribution Value"]
           .astype(str)
           .replace({",": ""}, regex=True)
           .astype(float)
    )
    df["Delivered_date"] = pd.to_datetime(
        df["Delivered_date"], errors="coerce", dayfirst=True
    )
    df["Month"] = df["Delivered_date"].dt.to_period("M").astype(str)
    df["Delivered Qty"] = df["Delivered Qty"].fillna(0)
    df["Total_Amount_Spent"] = df["Redistribution Value"] * df["Delivered Qty"]
    if "Order_Id" not in df.columns:
        df["Order_Id"] = (
            df["Customer_Phone"].astype(str)
            + "_"
            + df["Delivered_date"].dt.strftime("%Y%m%d%H%M%S")
            + "_"
            + df.groupby(["Customer_Phone", "Delivered_date"])  \
                  .cumcount().astype(str)
        )
    return df



# Streamlit App

st.set_page_config(page_title="AOP MVP", layout="wide")
st.title("📊 Annual Operation Plan: Actual vs. Industry Benchmarks")

# Sidebar: Industry benchmarks
st.sidebar.header("Industry Benchmark Targets")
bench_df = pd.DataFrame({
    "KPI": ["Annual Volume Growth", "Annual Value Growth"],
    "Default": [
        f"{BENCHMARKS['annual_volume_growth']*100:.1f}%",
        f"{BENCHMARKS['annual_value_growth']*100:.1f}%"
    ]
})
st.sidebar.table(bench_df)

# Sidebar: Target sliders
st.sidebar.header("Adjust Your Targets")
vol_growth = st.sidebar.slider(
    "Volume Growth Rate (%)", 0.0, 20.0,
    BENCHMARKS["annual_volume_growth"] * 100, step=0.5
) / 100.0
val_growth = st.sidebar.slider(
    "Value Growth Rate (%)", 0.0, 20.0,
    BENCHMARKS["annual_value_growth"] * 100, step=0.5
) / 100.0

# Load and clean the data
file_path = "/content/drive/MyDrive/data_sample_analysis_cleaned.csv"
df = load_clean_sales_data(file_path)

# Compute baseline totals & dynamic targets
baseline_qty = df["Delivered Qty"].sum()
baseline_val = df["Total_Amount_Spent"].sum()
target_qty   = baseline_qty * (1 + vol_growth)
target_val   = baseline_val * (1 + val_growth)

# KPI cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Delivered Qty (YTD)", f"{baseline_qty:,.0f}")
col2.metric("Target Qty",       f"{target_qty:,.0f}", delta=f"{vol_growth*100:.1f}%")
col3.metric("Value Spent (YTD)", f"₦{baseline_val:,.0f}")
col4.metric("Target Value",     f"₦{target_val:,.0f}", delta=f"{val_growth*100:.1f}%")

# Monthly trend chart (Actual vs. Monthly Target)
st.subheader("Monthly Actual vs. Target Qty")
monthly = df.groupby("Month", as_index=False)["Delivered Qty"].sum()
monthly["Target Qty"] = target_qty / len(monthly)
st.line_chart(monthly.set_index("Month")[['Delivered Qty','Target Qty']])

# Cumulative vs. Remaining Target
monthly["Cumulative Actual"]   = monthly["Delivered Qty"].cumsum()
monthly["Cumulative Target"]   = monthly["Target Qty"].cumsum()
monthly["Remaining Target"]    = target_qty - monthly["Cumulative Actual"]
st.subheader("Remaining Annual Target by Month")
st.line_chart(monthly.set_index("Month")[['Remaining Target']])

# Branch-level attainment
st.subheader("Branch-Level Attainment")
branch = df.groupby("Branch", as_index=False).agg({
    "Delivered Qty":      "sum",
    "Total_Amount_Spent": "sum"
})
branch['Qty Target']   = (branch['Delivered Qty'] / baseline_qty) * target_qty
branch['Value Target'] = (branch['Total_Amount_Spent'] / baseline_val) * target_val
branch['Qty Attainment']   = branch['Delivered Qty'] / branch['Qty Target']
branch['Value Attainment'] = branch['Total_Amount_Spent'] / branch['Value Target']

st.markdown("**Quantity Attainment by Branch**")
st.bar_chart(branch.set_index("Branch")["Qty Attainment"])
st.markdown("**Value Attainment by Branch**")
st.bar_chart(branch.set_index("Branch")["Value Attainment"])

# Top-10 SKU shortfalls with Brand
st.subheader("Top 10 SKUs by Qty Shortfall")
sku = (
    df.groupby(["Brand","SKU_Code"], as_index=False)["Delivered Qty"].sum()
)
sku['Target Qty'] = sku['Delivered Qty'] * (1 + vol_growth)
sku['Variance']   = sku['Delivered Qty'] - sku['Target Qty']
shortfalls = sku.nsmallest(10, 'Variance')[['Brand','SKU_Code','Variance']]
st.table(shortfalls)

# Salesperson performance
st.subheader("Salesperson Qty Attainment")
sales = (
    df.groupby("Salesman_Code", as_index=False)["Delivered Qty"].sum()
)
sales['Target Qty'] = sales['Delivered Qty'] * (1 + vol_growth)
sales['Attainment'] = sales['Delivered Qty'] / sales['Target Qty']
st.dataframe(sales.sort_values('Attainment', ascending=False))

# Run-rate projection
st.subheader("Run-Rate Projection")
avg_monthly = monthly['Delivered Qty'].mean()
projected   = avg_monthly * 12
st.write(
    f"If current monthly run-rate ({avg_monthly:,.0f} units) "  
    f"continues, projected annual Qty = {projected:,.0f} units."
)
