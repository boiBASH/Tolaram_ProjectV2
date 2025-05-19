import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from PIL import Image


# Must be first Streamlit command
st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_sales_data():
    df = pd.read_csv("data_sample_analysis.csv", encoding='latin1')
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
        "pred_next_date":      "Next Purchase Date",
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

if section == "📊 EDA Overview":
    st.subheader("Exploratory Data Analysis")
    tabs = st.tabs([
        "Brands Overview",
        "SKUs Overview",
        "Brand Deep Dive by SKU",
        "Customers Overview",
        "Overall Trends"
    ])

    # --- Brands Overview ---
    with tabs[0]:
        st.subheader("Brands Analysis")
        data_rev_brand = DF.groupby("Brand")["Redistribution Value"].sum().nlargest(10)
        st.markdown("**Top 10 Brands by Total Revenue**")
        st.bar_chart(data_rev_brand)
        data_qty_brand = DF.groupby("Brand")["Delivered Qty"].sum().nlargest(10)
        st.markdown("**Top 10 Brands by Quantity Sold**")
        st.bar_chart(data_qty_brand)
        share_brand = DF.groupby("Brand")["Delivered Qty"].sum() / DF["Delivered Qty"].sum() * 100
        st.markdown("**Brands by Share of Total Quantity (Top 10)**")
        st.bar_chart(share_brand.nlargest(10))
        df_brand_trend = DF.copy()
        df_brand_trend["MonthTS"] = df_brand_trend["Month"].dt.to_timestamp()
        top10_brands = df_brand_trend.groupby("Brand")["Delivered Qty"].sum().nlargest(10).index
        trend_brand = df_brand_trend[df_brand_trend["Brand"].isin(top10_brands)]
        trend_brand = trend_brand.groupby(["MonthTS", "Brand"])["Delivered Qty"].sum().unstack()
        st.markdown("**Monthly Quantity Trend for Top Brands**")
        st.line_chart(trend_brand)
        st.markdown("**Brand Pair Analysis (Top 5)**")
        df_brand_pairs = calculate_brand_pairs(DF).head(5)
        st.bar_chart(df_brand_pairs.set_index('Brand_Pair_Formatted')['Count'])
        brand_var = DF.groupby("Customer_Phone")["Brand"].nunique()
        dist_brand_var = brand_var.value_counts().sort_index().reset_index()
        dist_brand_var.columns = ['Unique Brands Purchased', 'Number of Customers']
        chart_brand_var = alt.Chart(dist_brand_var).mark_bar().encode(
            x=alt.X('Unique Brands Purchased:O', title='Unique Brands Purchased'),
            y=alt.Y('Number of Customers:Q', title='Number of Customers'),
            tooltip=['Unique Brands Purchased', 'Number of Customers']
        ).properties(title="Distribution of Brand Variety")
        st.altair_chart(chart_brand_var, use_container_width=True)

    # --- SKUs Overview ---
    with tabs[1]:
        st.subheader("SKUs Analysis")
        data_rev_sku = DF.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10)
        st.markdown("**Top 10 SKUs by Total Revenue**")
        st.bar_chart(data_rev_sku)
        data_qty_sku = DF.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10)
        st.markdown("**Top 10 SKUs by Quantity Sold**")
        st.bar_chart(data_qty_sku)
        share_sku = DF.groupby("SKU_Code")["Delivered Qty"].sum() / DF["Delivered Qty"].sum() * 100
        st.markdown("**SKUs by Share of Total Quantity (Top 10)**")
        st.bar_chart(share_sku.nlargest(10))
        df_sku_trend = DF.copy()
        df_sku_trend["MonthTS"] = df_sku_trend["Month"].dt.to_timestamp()
        top10_skus = df_sku_trend.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10).index
        trend_sku = df_sku_trend[df_sku_trend["SKU_Code"].isin(top10_skus)]
        trend_sku = trend_sku.groupby(["MonthTS", "SKU_Code"])["Delivered Qty"].sum().unstack()
        st.markdown("**Monthly Quantity Trend for Top SKUs**")
        st.line_chart(trend_sku)
        st.markdown("**Top 10 Most Frequently Bought SKU Pairs**")
        cnt_sku_pairs = Counter()
        df_sku_pairs = DF.copy()
        df_sku_pairs["Order_ID"] = df_sku_pairs["Customer_Phone"].astype(str) + "_" + df_sku_pairs["Delivered_date"].astype(str)
        for s in df_sku_pairs.groupby("Order_ID")["SKU_Code"].apply(set):
            if len(s) > 1:
                for pair in combinations(sorted(s), 2): cnt_sku_pairs[pair] += 1
        df_sku_pairs_top10 = pd.Series(cnt_sku_pairs).nlargest(10).to_frame(name="Count")
        df_sku_pairs_top10.index = df_sku_pairs_top10.index.map(lambda t: f"{t[0]} & {t[1]}")
        st.bar_chart(df_sku_pairs_top10)
        sku_var = DF.groupby("Customer_Phone")["SKU_Code"].nunique()
        dist_sku_var = sku_var.value_counts().sort_index()
        st.markdown("**Distribution of SKU Variety**")
        st.bar_chart(dist_sku_var)

    # --- Brand Deep Dive by SKU ---
    # --- Brand Deep Dive by SKU ---
    with tabs[2]:
        st.subheader("Brand Deep Dive by SKU")
        top_brands = DF.groupby("Brand")["Redistribution Value"].sum().nlargest(5).index.tolist()
        selected_brand_deep_dive = st.selectbox("Select a Brand for Deeper SKU Analysis:", top_brands)
        if selected_brand_deep_dive:
            brand_df = DF[DF['Brand'] == selected_brand_deep_dive]

            st.subheader(f"Brand Co-Purchases with {selected_brand_deep_dive}")
            df_pairs = calculate_brand_pairs(DF)

            # Filter pairs containing the selected brand
            filtered_pairs = df_pairs[
                df_pairs['Brand_Pair_Tuple'].apply(lambda x: selected_brand_deep_dive in x)
            ].sort_values(by='Count', ascending=False)

            
            if not filtered_pairs.empty:
                chart = alt.Chart(filtered_pairs).mark_bar().encode(
                    x=alt.X('Brand_Pair_Formatted', title='Co-Purchased Brand', sort='-y'),
                    y=alt.Y('Count', title='Frequency'),
                    color=alt.Color('Count', legend=alt.Legend(title='Frequency')),
                    tooltip=['Brand_Pair_Formatted', 'Count']
                ).properties(
                    width=600,
                    height=500,
                    title=f"Brands Purchased Alongside {selected_brand_deep_dive}"
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
                st.write(f"No co-purchases found for {selected_brand_deep_dive} with any other brand.")

            st.markdown(f"**Top 10 SKUs by Revenue in {selected_brand_deep_dive}**")
            top_skus_rev = brand_df.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10)
            st.bar_chart(top_skus_rev)

            st.markdown(f"**Top 10 SKUs by Quantity Sold in {selected_brand_deep_dive}**")
            top_skus_qty = brand_df.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10)
            st.bar_chart(top_skus_qty)

            df_sku_trend_brand = brand_df.copy()
            df_sku_trend_brand["MonthTS"] = df_sku_trend_brand["Month"].dt.to_timestamp()
            top5_skus_in_brand = df_sku_trend_brand.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(5).index
            trend_sku_brand = df_sku_trend_brand[df_sku_trend_brand["SKU_Code"].isin(top5_skus_in_brand)]
            trend_sku_brand = trend_sku_brand.groupby(["MonthTS", "SKU_Code"])["Delivered Qty"].sum().unstack()
            st.markdown(f"**Monthly Quantity Trend for Top 5 SKUs in {selected_brand_deep_dive}**")
            st.line_chart(trend_sku_brand)

                
    # --- Customers Overview ---
    with tabs[3]:
        st.subheader("Customers Analysis")
        st.markdown("**Top 10 Customers by Total Spending**")
        df_chart_cust_rev = DF.copy()
        df_chart_cust_rev['Customer_Info'] = df_chart_cust_rev['Customer_Name'] + ' (0' + df_chart_cust_rev['Customer_Phone'].astype(str) + ')'
        customer_ltv_with_name = (
            df_chart_cust_rev.groupby("Customer_Info")["Redistribution Value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        chart_cust_rev = alt.Chart(customer_ltv_with_name.reset_index()).mark_bar().encode(
            x=alt.X('Redistribution Value', title='Total Redistribution Value', axis=alt.Axis(format=',.2s')),
            y=alt.Y('Customer_Info', sort='-x', title='Customer'),
            tooltip=['Customer_Info', 'Redistribution Value']
        ).properties(height=500)
        st.altair_chart(chart_cust_rev, use_container_width=True)

        st.markdown("**Top 10 Buyers by Quantity Purchased**")
        df_chart_cust_qty = DF.copy()
        df_chart_cust_qty['Customer_Info'] = df_chart_cust_qty['Customer_Name'] + ' (0' + df_chart_cust_qty['Customer_Phone'].astype(str) + ')'
        top_buyers_qty_with_name = (
            df_chart_cust_qty.groupby("Customer_Info")["Delivered Qty"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        chart_cust_qty = alt.Chart(top_buyers_qty_with_name.reset_index()).mark_bar().encode(
            x=alt.X('Delivered Qty', title='Total Quantity', axis=alt.Axis(format=',.2s')),
            y=alt.Y('Customer_Info', sort='-x', title='Customer'),
            tooltip=['Customer_Info', 'Delivered Qty']
        ).properties(height=500)
        st.altair_chart(chart_cust_qty, use_container_width=True)

        st.markdown("**Buyer Type (Repeat vs. One-Time Buyers)**")
        counts_cust_type = DF.groupby("Customer_Phone")["Delivered_date"].nunique()
        summary_cust_type = (counts_cust_type == 1).map({True: "One-time", False: "Repeat"}).value_counts()
        st.bar_chart(summary_cust_type)

        st.markdown("**Monthly Purchase Value Trend for Top 5 Buyers**")
        df_cust_trend = DF.copy()
        df_cust_trend["MonthTS"] = df_cust_trend["Month"].dt.to_timestamp()
        top5_cust = df_cust_trend.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(5).index
        trend_cust = df_cust_trend[df_cust_trend["Customer_Phone"].isin(top5_cust)]
        trend_cust = trend_cust.groupby(["MonthTS", "Customer_Phone"])["Redistribution Value"].sum().unstack()
        st.line_chart(trend_cust)

        st.markdown("**Buyer Analysis (Top vs. Bottom in Latest Month)**")
        latest_month = DF['Month'].max()
        buyer_performance = DF[DF['Month'] == latest_month].groupby('Customer_Phone')['Redistribution Value'].sum()
        col_top_bottom = st.columns(2)
        with col_top_bottom[0]:
            st.write("Top 5 Buyers (Latest Month)")
            st.bar_chart(buyer_performance.nlargest(5))
        with col_top_bottom[1]:
            st.write("Bottom 5 Buyers (Latest Month)")
            st.bar_chart(buyer_performance.nsmallest(5))

    # --- Overall Trends ---
    with tabs[4]:
        st.subheader("Overall Sales Trends")
        monthly_summary_overall = DF.groupby("Month")[["Delivered Qty", "Redistribution Value"]].sum().reset_index()
        monthly_summary_overall["MonthTS"] = monthly_summary_overall["Month"].dt.to_timestamp()
        qty_overall = alt.Chart(monthly_summary_overall).mark_line(point=True).encode(
            x=alt.X("MonthTS:T", title="Month"),
            y=alt.Y("Delivered Qty:Q", axis=alt.Axis(title="Total Quantity", titleColor="royalblue")),
            color=alt.value("royalblue")
        )
        rev_overall = alt.Chart(monthly_summary_overall).mark_line(point=True).encode(
            x="MonthTS:T",
            y=alt.Y("Redistribution Value:Q", axis=alt.Axis(title="Total Revenue", titleColor="orange")),
            color=alt.value("orange")
        )
        dual_overall = alt.layer(qty_overall, rev_overall).resolve_scale(y="independent").properties(height=400)
        st.altair_chart(dual_overall, use_container_width=True)

        avg_order_value = DF.groupby("Month")["Redistribution Value"].mean().reset_index()
        avg_order_value["MonthTS"] = avg_order_value["Month"].dt.to_timestamp()
        chart_aov = alt.Chart(avg_order_value).mark_line(point=True).encode(
            x=alt.X("MonthTS:T", title="Month"),
            y=alt.Y("Redistribution Value:Q", title="Average Order Value")
        ).properties(title="Average Order Value Trend")
        st.altair_chart(chart_aov, use_container_width=True)

        ltv_top10 = DF.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(10).reset_index()
        ltv_top10.columns = ['Customer Phone', 'Total Spend']
        st.markdown("**Top 10 Customers by Lifetime Value (Total Spend)**")
        st.bar_chart(ltv_top10.set_index('Customer Phone')['Total Spend'])
elif section == "📉 Drop Detection":
    st.subheader("Brand-Level Month-over-Month (MoM) Revenue Drop Analysis")
    st.markdown(
        "This section analyzes the month-over-month percentage change in revenue for each brand to identify significant drops."
    )
    st.markdown(
        "**NB:\n**"
        "- Values in the table represent the MoM percentage change in revenue. \n"
        "- Upward trend is indicated by ⬆️, and downward trend by 🔻. \n"
        "- Previous month's revenue is shown in parentheses to provide context."
        , unsafe_allow_html=True
    )

    # --- 1. Data Preparation ---
    try:
        brand_month_revenue = DF.groupby(['Brand', 'Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    except KeyError as e:
        st.error(f"KeyError in data processing: {e}. Please ensure the 'Brand', 'Month', and 'Redistribution Value' columns are present in your data.")
        st.stop()

    # Calculate Month-over-Month (MoM) percentage change
    mom_change = brand_month_revenue.pct_change(axis=1) * 100

    # --- 2. Create Display DataFrame ---
    display_df = pd.DataFrame(index=brand_month_revenue.index)

    # Iterate through each month to format the display
    for i, col in enumerate(brand_month_revenue.columns):
        prev_month_col = brand_month_revenue.columns[i-1] if i > 0 else None
        mom_val = mom_change[col]
        prev_revenue = brand_month_revenue[prev_month_col] if prev_month_col is not None else pd.Series(index=brand_month_revenue.index)

        formatted_values = []
        for brand in brand_month_revenue.index:
            m = mom_val.get(brand)
            p = prev_revenue.get(brand)

            mom_str = f"{m:.1f}%" if pd.notna(m) else "Not Applicable"
            prev_str = f"{int(p):,}" if pd.notna(p) else "Not Applicable"
            arrow = ""
            if pd.notna(m):
                if m > 0:
                    arrow = "⬆️"
                elif m < 0:
                    arrow = "🔻"
            formatted_values.append(f"{mom_str}{arrow} ({prev_str})")

        display_df[f"{col}\n(Prev. Month\nRevenue)"] = formatted_values

    st.dataframe(display_df, use_container_width=True)

    # --- 3. Identify and Display Brands with Negative MoM Change ---
    negative_mom_changes = mom_change[mom_change < 0].stack().reset_index(name='MoM Change')
    if not negative_mom_changes.empty:
        st.subheader("Brands with Negative Month-over-Month Revenue Change")
        st.markdown(
            "The following table highlights brands that experienced a decrease in revenue compared to the previous month."
        )
        # Melt revenue data for merging
        melted_revenue = brand_month_revenue.melt(ignore_index=False, var_name='Month', value_name='Previous Month Revenue').reset_index()

        # Merge based on Brand and Month
        negative_brands_info = pd.merge(negative_mom_changes, melted_revenue, on=['Brand', 'Month'], how='left')

        negative_brands_info['MoM Change Formatted'] = negative_brands_info['MoM Change'].apply(lambda x: f"{x:.1f}%🔻" if pd.notna(x) else "Not Applicable")
        negative_brands_info['Previous Month Revenue Formatted'] = negative_brands_info['Previous Month Revenue'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "Not Applicable")

        display_negative = negative_brands_info[['Brand', 'Month', 'MoM Change Formatted', 'Previous Month Revenue Formatted']]
        display_negative.columns = ['Brand', 'Month', 'MoM Change', 'Previous Month Revenue']
        st.dataframe(display_negative, use_container_width=True, column_config={
            "MoM Change": st.column_config.Column(
                "MoM Change",
                help="Month-over-Month percentage change in revenue.",
            ),
            "Previous Month Revenue": st.column_config.Column(
                "Previous Month Revenue",
                help="Revenue in the previous month.",
            ),
        })
    else:
        st.info("No brands experienced a month-over-month revenue decrease in the selected period.")

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
