import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder # Import OneHotEncoder
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
    # Ensure 'Order_Id' is present for some calculations if missing
    if 'Order_Id' not in df.columns:
        df['Order_Id'] = df['Customer_Phone'].astype(str) + '_' + df['Delivered_date'].dt.strftime('%Y%m%d%H%M%S') + '_' + df.groupby(['Customer_Phone', 'Delivered_date']).cumcount().astype(str)
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
def analyze_customer_purchases_extended(df, customer_phone):
    customer_df = df[df['Customer_Phone'] == customer_phone].copy()

    if customer_df.empty:
        return f"No data found for customer phone: {customer_phone}"

    # Ensure date is sorted
    customer_df.sort_values('Delivered_date', inplace=True)

    # Add Month column
    customer_df['Month'] = customer_df['Delivered_date'].dt.to_period('M')

    # Basic customer information
    customer_name = customer_df['Customer_Name'].iloc[0] if not customer_df.empty else 'N/A'

    # 1. Total Brands Bought
    brands_bought = customer_df['Brand'].unique().tolist()
    total_brands_bought = len(brands_bought)

    # 2. SKUs of each Brand Bought
    brand_skus = customer_df.groupby('Brand')['SKU_Code'].unique().apply(list).to_dict()
    total_unique_skus_bought = customer_df['SKU_Code'].nunique()
    skus_bought = customer_df['SKU_Code'].unique().tolist()

    # 3. Purchase Summary by Brand
    purchase_summary_by_brand = {}
    for brand in brands_bought:
        brand_df = customer_df[customer_df['Brand'] == brand]
        last_purchase_date = brand_df['Delivered_date'].max().strftime('%Y-%m-%d') if not brand_df.empty else 'N/A'
        total_quantity = brand_df['Delivered Qty'].sum()
        total_spent = brand_df['Total_Amount_Spent'].sum()
        purchase_summary_by_brand[brand] = {
            'Last Purchase Date': last_purchase_date,
            'Total Quantity Bought': total_quantity,
            'Total Amount Spent': round(total_spent, 2)
        }

    # 4. Purchase Summary for Brand SKU
    purchase_summary_by_brand_sku = {}
    for brand, skus in brand_skus.items():
        purchase_summary_by_brand_sku[brand] = {}
        for sku in skus:
            sku_df = customer_df[(customer_df['Brand'] == brand) & (customer_df['SKU_Code'] == sku)]
            if not sku_df.empty:
                last_purchase = sku_df['Delivered_date'].max().strftime('%Y-%m-%d')
                monthly_qty_series = sku_df.groupby('Month')['Delivered Qty'].sum()
                avg_monthly_qty = monthly_qty_series.mean().round(2) if not monthly_qty_series.empty else 0.0
                monthly_spend_series = sku_df.groupby('Month')['Total_Amount_Spent'].sum()
                avg_monthly_spend = monthly_spend_series.mean().round(2) if not monthly_spend_series.empty else 0.0

                purchase_summary_by_brand_sku[brand][sku] = {
                    'Last Purchase Date': last_purchase,
                    'Avg Monthly Quantity': avg_monthly_qty,
                    'Avg Monthly Spend': avg_monthly_spend
                }
            else:
                purchase_summary_by_brand_sku[brand][sku] = {
                    'Last Purchase Date': 'N/A',
                    'Avg Monthly Quantity': 0.0,
                    'Avg Monthly Spend': 0.0
                }

    # 5. Salesman Analysis
    most_sold_salesman_info = 'N/A'
    salesman_designation = 'N/A'

    if 'Salesman_Name' in customer_df.columns and 'Order_Id' in customer_df.columns and not customer_df.empty:
        salesman_unique_order_counts = customer_df.groupby('Salesman_Name')['Order_Id'].nunique()

        if not salesman_unique_order_counts.empty and salesman_unique_order_counts.max() > 0:
            most_sold_salesman_name = salesman_unique_order_counts.idxmax()
            most_sold_salesman_count = salesman_unique_order_counts.max()
            most_sold_salesman_info = f"{most_sold_salesman_name} ({int(most_sold_salesman_count)} orders)"

            if 'Designation' in customer_df.columns and not customer_df[customer_df['Salesman_Name'] == most_sold_salesman_name].empty:
                salesman_designation = customer_df[customer_df['Salesman_Name'] == most_sold_salesman_name]['Designation'].iloc[0]
            else:
                salesman_designation = 'Designation data not available'

    # 6. Customer Branch
    customer_branch = 'N/A'
    if 'Branch' in customer_df.columns and not customer_df.empty:
        unique_branches = customer_df['Branch'].unique()
        if len(unique_branches) == 1:
            customer_branch = unique_branches[0]
        elif len(unique_branches) > 1:
            customer_branch = ", ".join(unique_branches)
        else:
            customer_branch = 'N/A (Branch data missing)'
    else:
        customer_branch = 'N/A (Branch column missing)'

    # 7. Total Order Count
    total_order_count = 0
    if 'Order_Id' in customer_df.columns:
        total_order_count = customer_df['Order_Id'].nunique()
    else:
        total_order_count = len(customer_df)

    report = {
        'Customer Phone': customer_phone,
        'Customer Name': customer_name,
        'Customer Branch': customer_branch,
        'Total Unique Brands Bought': total_brands_bought,
        'Brands Bought': brands_bought,
        'Total Order Count': total_order_count,
        'Top Salesperson': most_sold_salesman_info,
        'Salesperson Designation': salesman_designation,
        'Total Unique SKUs Bought': total_unique_skus_bought,
        'SKUs Bought': skus_bought,
        'Brand Level Summary': purchase_summary_by_brand,
        'Brand SKU Level Summary': purchase_summary_by_brand_sku,
        #'SKUs Grouped by Brand': brand_skus
    }

    return report

def predict_next_purchases(df_full, customer_phone):
    customer_df = df_full[df_full['Customer_Phone'] == customer_phone].copy()

    if customer_df.empty:
        return {
            'sku_predictions': pd.DataFrame(),
            'overall_next_brand_prediction': 'N/A',
        }

    customer_df['Delivered_date'] = pd.to_datetime(customer_df['Delivered_date'])
    customer_df.sort_values('Delivered_date', inplace=True)
    customer_df['Month'] = customer_df['Delivered_date'].dt.to_period('M')

    customer_df_sorted = customer_df.sort_values('Delivered_date')


    # --- SKU-Level Predictions ---
    last_purchase_date_sku = customer_df.groupby('SKU_Code')['Delivered_date'].max()

    avg_interval_days = {}
    for sku, grp in customer_df.groupby('SKU_Code'):
        dates = grp['Delivered_date'].drop_duplicates().sort_values()
        if len(dates) > 1:
            intervals = dates.diff().dt.days.dropna()
            if not intervals.empty:
                avg_interval_days[sku] = int(intervals.mean())
            else:
                avg_interval_days[sku] = np.nan
        else:
            avg_interval_days[sku] = np.nan

    avg_qty_sku = customer_df.groupby(['SKU_Code', 'Month'])['Delivered Qty'].sum().groupby('SKU_Code').mean().round(0)
    avg_spend_sku = customer_df.groupby(['SKU_Code', 'Month'])['Total_Amount_Spent'].sum().groupby('SKU_Code').mean().round(0)
    sku_to_brand = customer_df[['SKU_Code', 'Brand']].drop_duplicates().set_index('SKU_Code')['Brand']


    sku_predictions_df = pd.DataFrame({
        'Last Purchase Date': last_purchase_date_sku.dt.date,
        'Avg Interval Days': pd.Series(avg_interval_days),
        'Expected Quantity': avg_qty_sku,
        'Expected Spend': avg_spend_sku
    }).dropna(subset=['Avg Interval Days'])

    if not sku_predictions_df.empty:
        sku_predictions_df['Next Purchase Date'] = (
            pd.to_datetime(sku_predictions_df['Last Purchase Date']) +
            pd.to_timedelta(sku_predictions_df['Avg Interval Days'], unit='D')
        )
        sku_predictions_df = sku_predictions_df.merge(sku_to_brand.rename('Brand'), left_index=True, right_index=True, how='left')

        sku_predictions_df['Likely Purchase Date'] = sku_predictions_df['Next Purchase Date'].dt.strftime('%Y-%m-%d') + ' (' + sku_predictions_df['Next Purchase Date'].dt.day_name() + ')'

    else:
        sku_predictions_df['Next Purchase Date'] = pd.NA
        sku_predictions_df['Brand'] = pd.NA
        sku_predictions_df['Likely Purchase Date'] = pd.NA


    sku_predictions_df = sku_predictions_df.reset_index().rename(columns={
        'index': 'SKU Code',
        'Brand': 'Likely Brand',
    })
    sku_predictions_df = sku_predictions_df.sort_values(
        by='Next Purchase Date', ascending=True
    ).head(3) # Changed to .head(3)


    overall_next_brand_prediction = customer_df_sorted['Brand'].iloc[-1] if not customer_df_sorted.empty else 'N/A'


    return {
        'sku_predictions': sku_predictions_df,
        'overall_next_brand_prediction': overall_next_brand_prediction,
    }

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

# --- Load Data ---
DF = load_sales_data()
PRED_DF = load_model_preds()

# --- Recommender System Data Preparation (Cached) ---
@st.cache_data
def prepare_recommender_data(df_full):
    # Item-Item Collaborative Filtering (using Redistribution Value)
    user_item_matrix = df_full.pivot_table(index='Customer_Phone', columns='SKU_Code',
                                           values='Redistribution Value', aggfunc='sum', fill_value=0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity,
                                      index=user_item_matrix.columns,
                                      columns=user_item_matrix.columns)

    # Content-Based Filtering (using Brand and Branch)
    # Ensure 'Branch' column exists and handle potential missing values
    item_attributes_cols = ['SKU_Code', 'Brand']
    if 'Branch' in df_full.columns:
        item_attributes_cols.append('Branch')

    item_attributes = df_full[item_attributes_cols].drop_duplicates(subset=['SKU_Code']).set_index('SKU_Code')

    # Handle potential non-string types in 'Brand' or 'Branch' before OneHotEncoding
    for col in ['Brand', 'Branch']:
        if col in item_attributes.columns:
            item_attributes[col] = item_attributes[col].astype(str).fillna('Unknown')

    encoder = OneHotEncoder(handle_unknown='ignore') # handle_unknown='ignore' for new categories
    item_features_encoded = encoder.fit_transform(item_attributes)
    content_similarity = cosine_similarity(item_features_encoded)
    content_similarity_df = pd.DataFrame(content_similarity,
                                          index=item_attributes.index,
                                          columns=item_attributes.index)

    # Hybrid Similarity
    common_skus = item_similarity_df.index.intersection(content_similarity_df.index)
    if common_skus.empty:
        st.warning("No common SKUs found between collaborative and content-based models. Hybrid recommendations may not be possible.")
        return None, None, None # Return None if no common SKUs

    filtered_item_similarity = item_similarity_df.loc[common_skus, common_skus]
    filtered_content_similarity = content_similarity_df.loc[common_skus, common_skus]
    hybrid_similarity = (filtered_item_similarity + filtered_content_similarity) / 2

    # SKU to Brand mapping for recommendations display
    sku_brand_map = df_full[['SKU_Code', 'Brand']].drop_duplicates(subset='SKU_Code').set_index('SKU_Code')

    return user_item_matrix, hybrid_similarity, sku_brand_map

# --- Recommendation Functions ---
def recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, sku_brand_map, top_n=5):
    if customer_phone not in user_item_matrix.index:
        return pd.DataFrame() # Return empty if customer not in matrix

    purchased_skus = user_item_matrix.loc[customer_phone]
    purchased_skus = purchased_skus[purchased_skus > 0].index.tolist()

    if not purchased_skus:
        return pd.DataFrame() # Return empty if customer hasn't purchased anything

    # Filter hybrid_similarity to only include SKUs that are in the matrix
    valid_purchased_skus = [sku for sku in purchased_skus if sku in hybrid_similarity.columns]
    if not valid_purchased_skus:
        return pd.DataFrame() # No valid purchased SKUs for similarity calculation

    # Calculate scores for all SKUs based on purchased_skus
    sku_scores = hybrid_similarity[valid_purchased_skus].mean(axis=1)

    # Remove already purchased SKUs from recommendations
    sku_scores = sku_scores.drop(index=[s for s in purchased_skus if s in sku_scores.index], errors='ignore')

    if sku_scores.empty:
        return pd.DataFrame() # No recommendations if all are purchased or no scores

    top_skus = sku_scores.sort_values(ascending=False).head(top_n)

    # Ensure recommended SKUs exist in the sku_brand_map
    recommendations = sku_brand_map.loc[top_skus.index.intersection(sku_brand_map.index)].copy()
    recommendations['Similarity_Score'] = top_skus.loc[recommendations.index].values
    return recommendations.reset_index()

#def combined_report_recommender(customer_phone, user_item_matrix, hybrid_similarity, df_full, sku_brand_map, top_n=5):
    # Past Purchases
    #past_purchases = df_full[df_full['Customer_Phone'] == customer_phone][['SKU_Code', 'Brand']].drop_duplicates()
    #past_purchases['Type'] = 'Previously Purchased'
    #past_purchases['Similarity_Score'] = np.nan # No score for past purchases

    # Recommendations
    #recommendations = recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, sku_brand_map, top_n)
    #recommendations['Type'] = 'Recommended'

    # Combine and order columns
    #combined = pd.concat([past_purchases, recommendations[['SKU_Code', 'Brand', 'Similarity_Score', 'Type']]], ignore_index=True)
    #combined = combined[['Type', 'Brand', 'SKU_Code', 'Similarity_Score']] # Ensure consistent column order
    #return combined
def combined_report_recommender(customer_phone, user_item_matrix, hybrid_similarity, df_full, sku_brand_map, top_n=5):
    # Past Purchases
    past_purchases = df_full[df_full['Customer_Phone'] == customer_phone][['SKU_Code', 'Brand']].drop_duplicates()
    # No 'Type' or 'Similarity_Score' needed for this table

    # Recommendations
    recommendations = recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, sku_brand_map, top_n)
    # The 'Type' column can be added for internal filtering if needed, but not for display now.
    # The 'Similarity_Score' is already in 'recommendations' DataFrame.

    return past_purchases, recommendations # Return two separate DataFrames
# --- UI Setup ---
logo = Image.open("logo.png")
st.sidebar.image(logo, width=150)
st.sidebar.title("SALES DASHBOARD 🚀")
section = st.sidebar.radio(
    "Sections:",
    [
        "📊 EDA Overview",
        "📉 Drop Detection", 
        "👤 Customer Profiling",
        "🧑‍💻 Customer Profilling (Model Predictions)",
        "🔁 Cross-Selling", 
        "🔗 Brand Correlation",
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
            color=alt.Color('Delivered Qty'),
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
        "**NB:**"
        "\n- Values in the table represent the MoM percentage change in revenue. \n"
        "- Upward trend is indicated by ⬆️, and downward trend by🔻. \n"
        "- Previous month's revenue is shown in parentheses to provide context."
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

            mom_str = f"{m:.1f}%" if pd.notna(m) else "No Data"
            prev_str = f"{int(p):,}" if pd.notna(p) else "No Data"
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

        negative_brands_info['MoM Change Formatted'] = negative_brands_info['MoM Change'].apply(lambda x: f"{x:.1f}%🔻" if pd.notna(x) else "No Data")
        negative_brands_info['Previous Month Revenue Formatted'] = negative_brands_info['Previous Month Revenue'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "No Data")

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
        report = analyze_customer_purchases_extended(DF, cust)
        heuristic_predictions = predict_next_purchases(DF, cust) # Call the updated function

        if isinstance(report, str):
            st.write(report)
        else:
            st.markdown(f"**Customer Name:** {report['Customer Name']}")
            st.markdown(f"**Customer Branch:** {report['Customer Branch']}")
            st.markdown(f"**Total Unique Brands Bought:** {report['Total Unique Brands Bought']}")
            st.markdown(f"**Brands Bought:** {', '.join(report['Brands Bought'])}")
            st.markdown(f"**Total Order Count:** {report['Total Order Count']}")
            st.markdown(f"**Top Salesperson:** {report['Top Salesperson']}")
            st.markdown(f"**Salesperson Designation:** {report['Salesperson Designation']}")
            st.markdown(f"**Total Unique SKUs Bought:** {report['Total Unique SKUs Bought']}")
            st.markdown(f"**SKUs Bought:** {', '.join(report['SKUs Bought'])}")

            st.subheader("Brand Level Purchase Summary")
            brand_summary_df = pd.DataFrame.from_dict(report['Brand Level Summary'], orient='index')
            brand_summary_df = brand_summary_df.rename_axis('Brand').reset_index()
            st.dataframe(brand_summary_df, use_container_width=True)

            st.subheader("Brand SKU Level Purchase Summary")
            for brand, sku_summary in report['Brand SKU Level Summary'].items():
                st.markdown(f"**Brand:** {brand}")
                sku_summary_df = pd.DataFrame.from_dict(sku_summary, orient='index')
                sku_summary_df = sku_summary_df.rename_axis('SKU Code').reset_index()
                st.dataframe(sku_summary_df, use_container_width=True)

            # --- IMPROVED NEXT-PURCHASE PREDICTIONS DISPLAY ---
            st.subheader("Next Purchase Predictions (Heuristic)")
            st.markdown(f"**Overall Most Recently Bought Brand (for context):** {heuristic_predictions['overall_next_brand_prediction']}")

            st.markdown("---") # Separator for clarity

            st.markdown("**Predicted Next Purchases (SKU Level with Brand and Date/Day):**")
            if not heuristic_predictions['sku_predictions'].empty:
                # Display the combined SKU prediction table
                st.dataframe(
                    heuristic_predictions['sku_predictions'][[
                        'Likely Brand', 'SKU Code', 'Likely Purchase Date', 'Expected Quantity', 'Expected Spend'
                    ]],
                    use_container_width=True
                )
            else:
                st.info("Not enough historical data to provide detailed SKU purchase predictions for this customer.")

elif section == "🧑‍💻 Customer Profilling (Model Predictions)":
    st.subheader("Next-Purchase Model Predictions")
    cust = st.selectbox("Customer:", sorted(PRED_DF['Customer_Phone'].unique()))
    if cust:
        p = PRED_DF[PRED_DF['Customer_Phone'] == cust].drop(columns=['Customer_Phone']).set_index('SKU_Code')
        p['Probability'] = p['Probability'].map(lambda x: f"{x:.1f}%")
        st.dataframe(p, use_container_width=True)

# elif section == "🔁 Cross-Selling":
#     st.subheader("Brand Switching Patterns (Top 3)")
#     lp = DF.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
#     drop = lp[lp['Month'] < lp['Month'].max()]
#     sw = DF.merge(drop, on='Customer_Phone', suffixes=('','_dropped'))
#     sw = sw[(sw['Month'] > sw['Month_dropped']) & (sw['Brand'] != sw['Brand_dropped'])]
#     patterns = sw.groupby(['Brand_dropped','Brand']).size().reset_index(name='Count')
#     top3 = patterns.sort_values(['Brand_dropped','Count'], ascending=[True,False]).groupby('Brand_dropped').head(3)
#     st.dataframe(top3, use_container_width=True)

elif section == "🔗 Brand Correlation":
    st.subheader("Brand Correlation Matrix")
    mat = DF.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack(fill_value=0)
    st.dataframe(mat.corr().round(2), use_container_width=True)

elif section == "🤖 Recommender":
    st.subheader("Hybrid SKU Recommendations")
    st.markdown(
        "This section provides personalized SKU recommendations based on a hybrid approach, "
        "combining your past purchases with similar customers' behavior and item attributes."
    )

    # Prepare recommender data (cached)
    user_item_matrix, hybrid_similarity, sku_brand_map = prepare_recommender_data(DF)

    if user_item_matrix is None or hybrid_similarity is None or sku_brand_map is None:
        st.warning("Recommender system could not be initialized due to missing data or common SKUs.")
    else:
        # Get list of customers from the user-item matrix
        customer_list = sorted(user_item_matrix.index.tolist())
        sel_customer_recommender = st.selectbox("Select Customer for Recommendations:", customer_list)

        if st.button("Generate Recommendations"):
            if sel_customer_recommender:
                st.subheader(f"Recommendations for Customer {sel_customer_recommender}")

                # Call the updated function to get two DataFrames
                past_purchases_df, recommendations_df = combined_report_recommender(
                    sel_customer_recommender, user_item_matrix, hybrid_similarity, DF, sku_brand_map, top_n=5
                )

                # --- Display Previously Purchased SKUs ---
                st.markdown("### Previously Purchased SKUs")
                if not past_purchases_df.empty:
                    st.dataframe(past_purchases_df, use_container_width=True)
                else:
                    st.info("No past purchase data found for this customer.")

                st.markdown("---") # Separator

                # --- Display Recommended SKUs ---
                st.markdown("### Recommended SKUs")
                if not recommendations_df.empty:
                    st.dataframe(recommendations_df, use_container_width=True, column_config={
                        "Similarity_Score": st.column_config.NumberColumn(
                            "Similarity Score",
                            format="%.4f", # Format to 4 decimal places
                            help="Higher score indicates stronger similarity/recommendation."
                        )
                    })
                    st.info(
                        "A higher 'Similarity Score' indicates a stronger recommendation."
                    )
                else:
                    st.info("No new recommendations could be generated for this customer.")
            else:
                st.info("Please select a customer to generate recommendations.")
