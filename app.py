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
    df = pd.read_csv("cleaned_data_analysis.csv")
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
        "sku_predictions.csv",
        parse_dates=["last_purchase_date", "pred_next_date"],
    )
    preds = preds.rename(columns={
        "pred_next_date":     "Next Purchase Date",
        "pred_spend":          "Expected Spend",
        "pred_qty":            "Expected Quantity",
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
    avg_qty   = df.groupby(['SKU_Code','Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(0)
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
        "Top Revenue", "Top Quantity", "Buyer Types", "Buyer Trends",
        "SKU Trends", "Qty vs Revenue", "Avg Order Value", "Lifetime Value",
        "SKU Share %", "SKU Pairs", "SKU Variety", "Buyer Analysis"#, "Retention"
    ])
    # 1) Top Revenue
    with tabs[0]:
        data = DF.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10)
        st.markdown(f"**Top 10 SKUs by Total Revenue**")
        st.bar_chart(data)
    # 2) Top Quantity
    with tabs[1]:
        data = DF.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10)
        st.markdown(f"**Top 10 SKUs by Quantity Sold**")
        st.bar_chart(data)
    # 3) Buyer Types
    with tabs[2]:
        st.markdown(f"**Buyer Type (Repeat vs. One-Time Buyers)**")
        counts = DF.groupby("Customer_Phone")["Delivered_date"].nunique()
        summary = (counts == 1).map({True: "One-time", False: "Repeat"}).value_counts()
        st.bar_chart(summary)
    # 4) Buyer Trends
    with tabs[3]:
        st.markdown(f"**Monthly Purchase Value Trend for Top 5 Buyers**")
        df_b = DF.copy()
        df_b["MonthTS"] = df_b["Month"].dt.to_timestamp()
        top5 = df_b.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(5).index
        trend = df_b[df_b["Customer_Phone"].isin(top5)]
        trend = trend.groupby(["MonthTS","Customer_Phone"])["Redistribution Value"].sum().unstack()
        st.line_chart(trend)
    # 5) SKU Trends
    with tabs[4]:
        st.markdown(f"**Monthly Quantity Trend for Top 5 SKUs**")
        df_s = DF.copy()
        df_s["MonthTS"] = df_s["Month"].dt.to_timestamp()
        top5 = df_s.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(5).index
        trend = df_s[df_s["SKU_Code"].isin(top5)]
        trend = trend.groupby(["MonthTS","SKU_Code"])["Delivered Qty"].sum().unstack()
        st.line_chart(trend)
    # 6) Qty vs Revenue
    with tabs[5]:
        st.markdown(f"**Monthly Trend: Quantity vs. Revenue**")
        monthly_summary = DF.groupby("Month")[ ["Delivered Qty","Redistribution Value"] ].sum().reset_index()
        monthly_summary["MonthTS"] = monthly_summary["Month"].dt.to_timestamp()
        qty = alt.Chart(monthly_summary).mark_line(point=True).encode(
            x=alt.X("MonthTS:T", title="Month"),
            y=alt.Y("Delivered Qty:Q", axis=alt.Axis(title="Total Quantity", titleColor="royalblue")),
            color=alt.value("royalblue")
        )
        rev = alt.Chart(monthly_summary).mark_line(point=True).encode(
            x="MonthTS:T",
            y=alt.Y("Redistribution Value:Q", axis=alt.Axis(title="Total Revenue", titleColor="orange")),
            color=alt.value("orange")
        )
        dual = alt.layer(qty, rev).resolve_scale(y="independent").properties(height=400)
        st.altair_chart(dual, use_container_width=True)
    # 7) Avg Order Value
    with tabs[6]:
        st.markdown(f"**Top 10 Customers by Average Order Value**")
        data = DF.groupby("Customer_Phone")["Redistribution Value"].mean().nlargest(10)
        st.bar_chart(data)
    # 8) Lifetime Value
    with tabs[7]:
        st.markdown(f"**Top 10 Customers by Lifetime Value (Total Spend)**")
        data = DF.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(10)
        st.bar_chart(data)
    # 9) SKU Share %
    with tabs[8]:
        st.markdown(f"**Top 10 SKUs by Share of Total Quantity (in %)**")
        share = DF.groupby("SKU_Code")["Delivered Qty"].sum() / DF["Delivered Qty"].sum() * 100
        st.bar_chart(share.nlargest(10))
    # 10) SKU Pairs
    with tabs[9]:
        st.markdown(f"**Top 10 Most Frequently Bought SKU Pairs**")
        cnt = Counter()
        df_p = DF.copy()
        df_p["Order_ID"] = df_p["Customer_Phone"].astype(str) + "_" + df_p["Delivered_date"].astype(str)
        for s in df_p.groupby("Order_ID")["SKU_Code"].apply(set):
            if len(s) > 1:
                for pair in combinations(sorted(s), 2): cnt[pair] += 1
        df_pairs = pd.Series(cnt).nlargest(10).to_frame(name="Count")
        df_pairs.index = df_pairs.index.map(lambda t: f"{t[0]} & {t[1]}")
        st.bar_chart(df_pairs)
    # 11) SKU Variety
    with tabs[10]:
        st.markdown(f"**Distribution of SKU Variety Per Customer (Number os customer by number of unique SKUs purchased)**")
        sku_var = DF.groupby("Customer_Phone")["SKU_Code"].nunique()
        dist = sku_var.value_counts().sort_index()
        st.bar_chart(dist)
    # 12) Buyer Analysis
    with tabs[11]:
        st.markdown(f"**Buyer Analysis (Top Buyers and Button Buyers)**")
        mm = DF['Month'].max()
        bd = DF[DF['Month']==mm].groupby('Customer_Phone')['Redistribution Value'].sum()
        st.write("Top Buyers (Latest Month)")
        st.bar_chart(bd.nlargest(10))
        st.write("Bottom Buyers (Latest Month)")
        st.bar_chart(bd.nsmallest(10))
    # 13) Retention
    #with tabs[12]:
    #    st.markdown(f"**Retention**")
    #    orders = DF.groupby('Month')['Order_Id'].nunique()
    #    st.line_chart(orders.rolling(3).mean())

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
    # item content/features
    pf = pd.get_dummies(
        DF[['SKU_Code','Brand']].drop_duplicates(), columns=['Brand']
    ).set_index('SKU_Code')
    # compute similarities
    user_sim = cosine_similarity(uim)
    item_sim = cosine_similarity(pf)
    user_sim_df = pd.DataFrame(user_sim, index=uim.index, columns=uim.index)
    item_sim_df = pd.DataFrame(item_sim, index=pf.index, columns=pf.index)

    sel = st.selectbox("Select Customer", uim.index)
    if st.button("Recommend"):
        # collaborative score: weighted sum of user similarities
        collab_scores = uim.T.dot(user_sim_df[sel])
        # remove SKUs already purchased by sel
        purchased = uim.loc[sel][uim.loc[sel] > 0].index
        collab_scores = collab_scores.drop(index=purchased, errors='ignore')
        # content score: sum of item similarities to purchased SKUs
        content_scores = item_sim_df.loc[purchased].sum(axis=0)
        content_scores = content_scores.drop(index=purchased, errors='ignore')
        # combine with equal weight
        combined = 0.5 * collab_scores + 0.5 * content_scores
        top5 = combined.nlargest(5)
        result = top5.reset_index()
        result.columns = ['SKU_Code', 'Score']
        st.dataframe(result, use_container_width=True)
