import streamlit as st
# Must be first Streamlit command
st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")

import pandas as pd
import numpy as np
import altair as alt
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_sales_data():
    df = pd.read_csv("data_sample_analysis.csv", encoding='latin-1')
    df['Redistribution Value'] = (
        df['Redistribution Value']
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    df['Delivered_date'] = pd.to_datetime(
        df['Delivered_date'], errors='coerce', dayfirst=True
    )
    df['Month'] = df['Delivered_date'].dt.to_period('M')
    df['Delivered Qty'] = df['Delivered Qty'].fillna(0)
    df['Total_Amount_Spent'] = df['Redistribution Value'] * df['Delivered Qty']
    return df

@st.cache_data
def load_model_preds():
    preds = pd.read_csv(
        "purchase_predictions_major.csv",
        parse_dates=["last_purchase_date", "pred_next_date"],
    )
    preds = preds.rename(columns={
        "pred_next_brand":     "Next Brand Purchase",
        "pred_next_date":     "Next Purchase Date",
        "pred_spend":         "Expected Spend",
        "pred_qty":         "Expected Quantity",
        "probability":         "Probability"
    })
    preds["Next Purchase Date"] = preds["Next Purchase Date"].dt.date
    preds["Expected Spend"] = preds["Expected Spend"].round(0).astype(int)
    preds["Expected Quantity"] = preds["Expected Quantity"].round(0).astype(int)
    preds["Probability"] = (preds["Probability"] * 100).round(1)
    def suggest(p):
        if p >= 70:
            return "Follow-up/Alert"
        if p >= 50:
            return "Cross Sell"
        return "Discount"
    preds["Suggestion"] = preds["Probability"].apply(suggest)
    return preds

# --- Heuristic Profiling Functions ---
def analyze_customer_purchases(customer_phone):
    df = DF[DF['Customer_Phone'] == customer_phone].copy()
    if df.empty:
        return {}
    df.sort_values('Delivered_date', inplace=True)
    skus = df['SKU_Code'].unique().tolist()
    last_purchase = df.groupby('SKU_Code')['Delivered_date'].max().dt.strftime('%Y-%m-%d').to_dict()
    monthly_qty = df.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(2).to_dict()
    avg_interval = {}
    for sku, grp in df.groupby('SKU_Code'):
        dates = grp['Delivered_date'].drop_duplicates().sort_values()
        if len(dates) > 1:
            avg_interval[sku] = round((dates.diff().dt.days.dropna() / 30.44).mean(), 2)
        else:
            avg_interval[sku] = 'One'
    monthly_spend = df.groupby(['SKU_Code','Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(2).to_dict()
    report = {
        'Customer Phone': customer_phone,
        'Total Unique SKUs Bought': len(skus),
        'SKUs Bought': skus,
        'Purchase Summary by SKU': {}
    }
    for sku in skus:
        report['Purchase Summary by SKU'][sku] = {
            'Last Purchase Date': last_purchase.get(sku, 'N/A'),
            'Avg Monthly Quantity': monthly_qty.get(sku, 0),
            'Avg Purchase Interval (Months)': avg_interval.get(sku, 'N/A'),
            'Avg Monthly Spend': monthly_spend.get(sku, 0)
        }
    return report

def predict_next_purchases(customer_phone):
    df = DF[DF['Customer_Phone'] == customer_phone].copy()
    if df.empty:
        return pd.DataFrame()
    last_purchase = df.groupby('SKU_Code')['Delivered_date'].max()
    avg_interval_days = {}
    for sku, grp in df.groupby('SKU_Code'):
        dates = grp['Delivered_date'].drop_duplicates().sort_values()
        if len(dates) > 1:
            avg_interval_days[sku] = int(dates.diff().dt.days.dropna().mean())
        else:
            avg_interval_days[sku] = np.nan
    avg_qty     = df.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(0)
    avg_spend = df.groupby(['SKU_Code','Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(0)
    score_df = pd.DataFrame({
        'Last Purchase Date': last_purchase.dt.date,
        'Avg Interval Days': pd.Series(avg_interval_days),
        'Expected Quantity': avg_qty,
        'Expected Spend': avg_spend
    }).dropna(subset=['Avg Interval Days'])
    score_df['Next Purchase Date'] = (
        pd.to_datetime(score_df['Last Purchase Date']) + pd.to_timedelta(score_df['Avg Interval Days'], unit='D')
    ).dt.date
    return score_df.sort_values('Avg Interval Days').head(3)[['Next Purchase Date','Expected Spend','Expected Quantity']]

# --- Load Data ---
DF = load_sales_data()
PRED_DF = load_model_preds()

# --- Utility function ---
def calculate_brand_pairs(df):
    """
    Calculates the frequency of brand pairs appearing together in orders.

    Args:
        df (pd.DataFrame): The input DataFrame with sales data, containing columns
                           'Customer_Phone', 'Delivered_date', and 'Brand'.

    Returns:
        pd.DataFrame: A DataFrame containing the top brand pairs and their
                      co-occurrence counts, formatted for display.
    """
    df['Order_ID'] = df['Customer_Phone'].astype(str) + "_" + df['Delivered_date'].astype(str)
    order_brands = df.groupby("Order_ID")["Brand"].apply(set)

    pair_counts = Counter()
    for items in order_brands:
        if len(items) > 1:
            for pair in combinations(items, 2):
                pair_counts[tuple(sorted(pair))] += 1

    pair_df = pd.DataFrame(pair_counts.items(), columns=["Brand_Pair_Tuple", "Count"]).sort_values(by="Count", ascending=False)
    pair_df['Brand_Pair_Formatted'] = pair_df['Brand_Pair_Tuple'].apply(lambda x: f"{x[0]} & {x[1]}")
    return pair_df

# --- UI Setup ---
logo = Image.open("logo.png")
st.sidebar.image(logo, width=80)
st.sidebar.title("🚀 Sales Insights")
section = st.sidebar.radio(
    "Select Section:",
    [
        "📊 EDA Overview", "📉 Drop Detection", "👤 Customer Profiling",
        "👤 Customer Profilling (Model Predictions)", "🔁 Cross-Selling", "🔗 Brand Correlation",
        "🤖 Recommender"
    ]
)
st.title("📊 Sales Intelligence Dashboard")

# --- EDA Overview ---
if section == "📊 EDA Overview":
    st.subheader("Exploratory Data Analysis")
    tabs = st.tabs([
        "Top Revenue by Brand", "Top Revenue by SKU", "Top Quantity by Brand", "Top Quantity by SKU", "Top Customers", "Top Buyers by Quantity", "Buyer Types", "Buyer Trends",
        "Brand Trends", "SKU Trends", "Qty vs Revenue", "Avg Order Value", "Lifetime Value",
        "SKU Share %", "SKU Pairs", "SKU Variety", "Buyer Analysis", "Brand Pairs" #  moved tab and added Brand Pairs
        #, "Retention"
    ])

    # ... (rest of the EDA Overview code) ...

    # 14) Brand Pairs
    with tabs[17]:
        st.markdown(f"**Brand Pair Analysis**")
        df_pairs = calculate_brand_pairs(DF)  # Use the function

        all_brands = sorted(list(set(brand for pair in df_pairs['Brand_Pair_Tuple'] for brand in pair)))
        selected_brand = st.selectbox("Select a Brand to Analyze:", all_brands)

        # Filter pairs containing the selected brand
        filtered_pairs = df_pairs[
            df_pairs['Brand_Pair_Tuple'].apply(lambda x: selected_brand in x)
        ].sort_values(by='Count', ascending=False)

        if not filtered_pairs.empty:
            st.subheader(f"Brand Pairs for {selected_brand}")
            chart = alt.Chart(filtered_pairs).mark_bar().encode(
                x=alt.X('Brand_Pair_Formatted', title='Brand Pair'),
                y=alt.Y('Count', title='Frequency'),
                color=alt.Color('Count', legend=alt.Legend(title='Frequency')),
                tooltip=['Brand_Pair_Formatted', 'Count']
            ).properties(
                width=600,
                height=400,
                title=f"Brand Pairs for {selected_brand}"
            )
            text = chart.mark_text(
                align='center',
                baseline='bottom',
                dy=-5
            ).encode(
                text='Count',
                color=alt.value('black')
            )
            final_chart = chart + text
            st.altair_chart(final_chart, use_container_width=True)
        else:
            st.write(f"No co-purchases found for {selected_brand} with any other brand.")

        # Display Top 5 Brand Pairs Overall
        st.subheader("Top 5 Most Frequently Bought Brand Pairs Overall")
        top_5_pairs = df_pairs.head(5)
        chart_top_5 = alt.Chart(top_5_pairs).mark_bar().encode(
            x=alt.X('Brand_Pair_Formatted', title='Brand Pair'),
            y=alt.Y('Count', title='Frequency'),
            color=alt.Color('Count', legend=alt.Legend(title='Frequency')),
            tooltip=['Brand_Pair_Formatted', 'Count']
        ).properties(
            width=600,
            height=400,
            title="Top 5 Most Frequently Bought Brand Pairs Overall"
        )
        text_top_5 = chart_top_5.mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            text='Count',
            color=alt.value('black')
        )
        final_chart_top_5 = chart_top_5 + text_top_5
        st.altair_chart(final_chart_top_5, use_container_width=True)

# --- Other Sections ---
elif section == "📉 Drop Detection":
    st.subheader("Brand-Level MoM Drop (>30%)")
    bm = DF.groupby(['Brand','Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    mom = bm.pct_change(axis=1) * 100
    flags = mom < -30
    disp = mom.round(1).astype(str)
    disp[flags] += "% 🔻"
    disp[~flags] = ""
    st.dataframe(disp)

elif section == "👤 Customer Profiling":
    st.subheader("Customer Purchase Deep-Dive")
    cust = st.selectbox("Select Customer Phone:", sorted(DF['Customer_Phone'].unique()))
    if cust:
        rep = analyze_customer_purchases(cust)
        st.markdown(f"**Total Unique SKUs Bought:** {rep['Total Unique SKUs Bought']}")
        st.markdown(f"**SKUs Bought:** {', '.join(rep['SKUs Bought'])}")
        sku_df = pd.DataFrame.from_dict(rep['Purchase Summary by SKU'], orient='index')
        sku_df = sku_df.rename_axis('SKU_Code').reset_index()
        st.dataframe(sku_df, use_container_width=True)
        st.subheader("Next-Purchase Predictions (Heuristic)")
        st.dataframe(predict_next_purchases(cust), use_container_width=True)

elif section == "👤 Customer Profilling (Model Predictions)":
    st.subheader("Next-Purchase Model Predictions")
    cust = st.selectbox("Customer:", sorted(PRED_DF['Customer_Phone'].unique()))
    if cust:
        p = PRED_DF[PRED_DF['Customer_Phone'] == cust].drop(columns=['Customer_Phone']).set_index('SKU_Code')
        p['Probability'] = p['Probability'].map(lambda x: f"{x:.1f}%")
        st.dataframe(p, use_container_width=True)

elif section == "🔁 Cross-Selling":
    st.subheader("Brand Switching Patterns (Top 3)")
    lp = DF.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
    drop = lp[lp['Month'] < lp['Month'].max()]
    sw = DF.merge(drop, on='Customer_Phone', suffixes=('','_dropped'))
    sw = sw[(sw['Month'] > sw['Month_dropped']) & (sw['Brand'] != sw['Brand_dropped'])]
    patterns = sw.groupby(['Brand_dropped','Brand']).size().reset_index(name='Count')
    top3 = patterns.sort_values(['Brand_dropped','Count'], ascending=[True,False]).groupby('Brand_dropped').head(3)
    st.dataframe(top3, use_container_width=True)

elif section == "🔗 Brand Correlation":
    st.subheader("Brand Correlation Matrix")
    mat = DF.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack(fill_value=0)
    st.dataframe(mat.corr().round(2), use_container_width=True)

elif section == "🤖 Recommender":
    st.subheader("Hybrid SKU Recommendations")
    # user-item interaction matrix
    uim = DF.pivot_table(
        index='Customer_Phone', columns='SKU_Code',
        values='Redistribution Value', aggfunc='sum'
    ).fillna(0)
  
