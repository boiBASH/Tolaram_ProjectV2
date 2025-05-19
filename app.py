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

    # 1) Top Revenue by SKU
    with tabs[0]:
        data = DF.groupby("Brand")["Redistribution Value"].sum().nlargest(10)
        st.markdown(f"**Top 10 Brand by Total Revenue**")
        st.bar_chart(data)
    # 1) Top Revenue by SKU
    with tabs[1]:
        data = DF.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10)
        st.markdown(f"**Top 10 SKUs by Total Revenue**")
        st.bar_chart(data)
    # Top 10 Brand by quantity sold
    with tabs[2]:
        data = DF.groupby("Brand")["Delivered Qty"].sum().nlargest(10)
        st.markdown(f"**Top 10 Brand by Quantity Sold**")
        st.bar_chart(data)
        
    # 2) Top 10 Quantity by SKU
    with tabs[3]:
        data = DF.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10)
        st.markdown(f"**Top 10 SKUs by Quantity Sold**")
        st.bar_chart(data)

    # Top 10 Customers by Total Spending
    with tabs[4]:
        st.markdown(f"**Top 10 Customers by Total Spending**")
        # Preprocess the data as in the original code
        df_chart = DF.copy()
        df_chart['Customer_Info'] = df_chart['Customer_Name'] + ' (0' + df_chart['Customer_Phone'].astype(str) + ')'
        customer_ltv_with_name = (
            df_chart.groupby("Customer_Info")["Redistribution Value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        # Use Altair for the bar chart
        chart = alt.Chart(customer_ltv_with_name.reset_index()).mark_bar().encode(
            x=alt.X('Redistribution Value', title='Total Redistribution Value', axis=alt.Axis(format=',.2s')), # Added formatting
            y=alt.Y('Customer_Info', sort='-x', title='Customer Name and Phone Number'),
            color=alt.Color('Redistribution Value'),#, scale=alt.Scale(range='heatmap')), # changed the scale
            tooltip=['Customer_Info', 'Redistribution Value']
        ).properties(
            #title='Top 10 Customers by Total Spending',
            width=600,  # Increased width
            height=750 # increased height
        )

        # Add text labels to the bars
        text = chart.mark_text(
            align='left',
            baseline='middle',
            dx=5,  # Nudge labels to the right
            dy=0  # Center labels vertically
        ).encode(
            text=alt.Text('Redistribution Value', format=',.2s'), # Added formatting
            color=alt.value('black')  # Set the color of the labels to black
        )

        # Combine the bars and labels
        final_chart = chart + text

        # Display the chart in Streamlit
        st.altair_chart(final_chart, use_container_width=True)

    # New Tab: Top Buyers by Quantity
    with tabs[5]:
        st.markdown(f"**Top 10 Buyers by Quantity Purchased**")
        # Preprocess the data as in the original code
        df_chart_qty = DF.copy()
        df_chart_qty['Customer_Info'] = df_chart_qty['Customer_Name'] + ' (0' + df_chart_qty['Customer_Phone'].astype(str) + ')'
        top_buyers_qty_with_name = (
            df_chart_qty.groupby("Customer_Info")["Delivered Qty"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        # Use Altair for the bar chart
        chart_qty = alt.Chart(top_buyers_qty_with_name.reset_index()).mark_bar().encode(
            x=alt.X('Delivered Qty', title='Total Quantity', axis=alt.Axis(format=',.2s')),
            y=alt.Y('Customer_Info', sort='-x', title='Customer Name and Phone Number'),
            color=alt.Color('Delivered Qty'),
            tooltip=['Customer_Info', 'Delivered Qty']
        ).properties(
            width=600,
            height=750
        )
        
        # Add text labels to the bars
        text_qty = chart_qty.mark_text(
            align='left',
            baseline='middle',
            dx=5,
        ).encode(
            text=alt.Text('Delivered Qty', format=',.2s'),
            color=alt.value('black')
        )
        
        final_chart_qty = chart_qty + text_qty
        st.altair_chart(final_chart_qty, use_container_width=True)

    # 3) Buyer Types
    with tabs[6]:
        st.markdown(f"**Buyer Type (Repeat vs. One-Time Buyers)**")
        counts = DF.groupby("Customer_Phone")["Delivered_date"].nunique()
        summary = (counts == 1).map({True: "One-time", False: "Repeat"}).value_counts()
        st.bar_chart(summary)
    # 4) Buyer Trends
    with tabs[7]:
        st.markdown(f"**Monthly Purchase Value Trend for Top Buyers**")
        df_b = DF.copy()
        df_b["MonthTS"] = df_b["Month"].dt.to_timestamp()
        top5 = df_b.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(5).index
        trend = df_b[df_b["Customer_Phone"].isin(top5)]
        trend = trend.groupby(["MonthTS","Customer_Phone"])["Redistribution Value"].sum().unstack()
        st.line_chart(trend)

        # 5) SKU Trends
    with tabs[8]:
        st.markdown(f"**Monthly Quantity Trend for Top Brand**")
        df_s = DF.copy()
        df_s["MonthTS"] = df_s["Month"].dt.to_timestamp()
        top10 = df_s.groupby("Brand")["Delivered Qty"].sum().nlargest(10).index
        trend = df_s[df_s["Brand"].isin(top10)]
        trend = trend.groupby(["MonthTS","Brand"])["Delivered Qty"].sum().unstack()
        st.line_chart(trend)
        
    # 5) SKU Trends
    with tabs[9]:
        st.markdown(f"**Monthly Quantity Trend for Top SKUs**")
        df_s = DF.copy()
        df_s["MonthTS"] = df_s["Month"].dt.to_timestamp()
        top10 = df_s.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10).index
        trend = df_s[df_s["SKU_Code"].isin(top10)]
        trend = trend.groupby(["MonthTS","SKU_Code"])["Delivered Qty"].sum().unstack()
        st.line_chart(trend)
    # 6) Qty vs Revenue
    with tabs[10]:
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
    with tabs[11]:
        st.markdown(f"**Top 10 Customers by Average Order Value**")
        data = DF.groupby("Customer_Phone")["Redistribution Value"].mean().nlargest(10)
        st.bar_chart(data)
    # 8) Lifetime Value
    with tabs[12]:
        st.markdown(f"**Top 10 Customers by Lifetime Value (Total Spend)**")
        data = DF.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(10)
        st.bar_chart(data)
    # 9) SKU Share %
    with tabs[13]:
        st.markdown(f"**Top 10 SKUs by Share of Total Quantity (in %)**")
        share = DF.groupby("SKU_Code")["Delivered Qty"].sum() / DF["Delivered Qty"].sum() * 100
        st.bar_chart(share.nlargest(10))
    # 10) SKU Pairs
    with tabs[14]:
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
    with tabs[15]:
        st.markdown(f"**Distribution of SKU Variety Per Customer (Number os customer by number of unique SKUs purchased)**")
        sku_var = DF.groupby("Customer_Phone")["SKU_Code"].nunique()
        dist = sku_var.value_counts().sort_index()
        st.bar_chart(dist)
    # 13) Buyer Analysis #this used to be 12, now is 13
    with tabs[16]:
        st.markdown(f"**Buyer Analysis (Top Buyers and Button Buyers)**")
        mm = DF['Month'].max()
        bd = DF[DF['Month']==mm].groupby('Customer_Phone')['Redistribution Value'].sum()
        st.write("Top Buyers (Latest Month)")
        st.bar_chart(bd.nlargest(10))
        st.write("Bottom Buyers (Latest Month)")
        st.bar_chart(bd.nsmallest(10))
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
                x=alt.X('Brand_Pair_Formatted', title='Brand Pair', sort='-y'),  # Sort by descending y (Count)
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
            x=alt.X('Brand_Pair_Formatted', title='Brand Pair', sort='-y'), # Sort by descending y (Count)
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
    st.subheader("Brand-Level MoM Drop Analysis")

    # --- 1. Data Preparation ---
    # Group data by Brand and Month, calculate total revenue

    
    try:
        brand_month_revenue = DF.groupby(['Brand', 'Month'])['Redistribution Value'].sum().unstack(fill_value=0)
    except KeyError as e:
        st.error(f"KeyError in data processing: {e}.  Please ensure the 'Brand', 'Month', and 'Redistribution Value' columns are present in your data.")
        st.stop()



    # Calculate Month-over-Month (MoM) percentage change
    mom_change = brand_month_revenue.pct_change(axis=1) * 100
    
    # --- 2. Identify Significant Drops ---
    # No threshold, show all changes
    significant_changes = mom_change # Removed the drop_threshold
    
    # --- 3. Create Display DataFrame ---
    # Prepare data for display, including MoM change and drop flag
    display_data = mom_change.round(1).astype(str) + "%"  # Format as string with %
    
    def get_arrow(value):
        if pd.isna(value):  # Check for NaN
            return ""
        elif value > 0:
            return " ⬆️"  # Up arrow
        elif value < 0:
            return " 🔻"  # Down arrow
        else:
            return ""       # No arrow for zero change

    display_data_with_arrows = mom_change.round(1).applymap(lambda x: f"{x:.1f}%{get_arrow(x)}")
    
    # --- 4.  Add Previous Month Revenue for Context
    # Create a DataFrame to hold the previous month's revenue for each brand
    prev_month_revenue = brand_month_revenue.shift(axis=1).fillna(0)  # Shift by one month, fill NaNs with 0
    
    # Format the previous month's revenue for display
    prev_month_display = prev_month_revenue.round(0).astype(str)
    
    # Combine the MoM change and previous month revenue for display
    display_df = display_data_with_arrows.copy()  # Create a copy to avoid modifying the original
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str) + '(' + prev_month_display[col].astype(str) + ')'
    
    # Rename columns for better display
    display_df = display_df.rename(columns={col: f"{col}\n(Prev. Month\nRevenue)" for col in display_df.columns})
    
    # --- 5. Streamlit Output ---
    st.write(
        "This table shows the Month-over-Month (MoM) percentage change in revenue for each brand.  "
        "Up (⬆️) and down (🔻) arrows indicate the direction of change.  "
        "Previous month's revenue is shown in parentheses."
    )
    st.dataframe(display_df, use_container_width=True) #make it fill width

    # --- 6.  Additional Analysis and Insights (Optional) ---
    # You could add more detailed analysis here, such as:
    # -  Number of brands with drops in the selected period.
    # -  Average drop percentage.
    # -  Brands with consecutive drops.
    # -  Correlation with other factors (e.g., promotions, seasonality).

    # Example of additional output:
    num_drops = (mom_change < 0).sum().sum() #changed to show all negative changes
    st.write(f"Total number of brands with negative MoM change: {num_drops}")
    
    if num_drops > 0:
        st.write("Brands with negative MoM change (including previous month revenue and MoM % change):")
        
        # Melt the mom_change DataFrame to long format to get the MoM change values
        mom_long = mom_change.reset_index().melt(id_vars='Brand', var_name='Month', value_name='MoM Change')
        
        # Melt previous month revenue to long format
        prev_month_long = prev_month_revenue.reset_index().melt(id_vars='Brand', var_name='Month', value_name='Previous Month Revenue')

        # Merge
        brands_with_changes = pd.merge(mom_long, prev_month_long, on=['Brand', 'Month'])
        
        # Filter for negative MoM change
        brands_with_changes = brands_with_changes[brands_with_changes['MoM Change'] < 0].copy()
        
        # Explicitly set data types to avoid Arrow conversion issues
        brands_with_changes['Brand'] = brands_with_changes['Brand'].astype(str)
        brands_with_changes['Month'] = brands_with_changes['Month'].astype(str)
        brands_with_changes['MoM Change'] = brands_with_changes['MoM Change'].astype(float) #important
        brands_with_changes['Previous Month Revenue'] = brands_with_changes['Previous Month Revenue'].astype(float)
        
        # Format the MoM Change for display, add arrow
        brands_with_changes['MoM Change'] = brands_with_changes['MoM Change'].apply(lambda x: f"{x:.1f}%{get_arrow(x)}")
        
        # Display the DataFrame
        st.dataframe(brands_with_changes[['Brand', 'Month', 'MoM Change', 'Previous Month Revenue']], use_container_width=True)
        
    else:
        st.write("No brands with negative MoM change.")
    

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
