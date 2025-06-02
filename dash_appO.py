import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import base64 # For embedding logo

# --- Data Loading & Preprocessing ---
# In a Dash app, it's common to load data once globally if it's static
# or use a more sophisticated data loading strategy for very large datasets.
# For simplicity, we'll load it once here.

# Assuming 'data_sample_analysis.csv' is in the same directory as this script
# and 'purchase_predictions_major.csv' as well.
# Also assuming 'logo.png' is available.

def load_sales_data():
    """
    Loads sales data from CSV, performs necessary type conversions and feature engineering.
    Includes low_memory=False to handle potential mixed types in columns more robustly.
    """
    df = pd.read_csv("data_sample_analysis.csv", encoding='latin1', low_memory=False)

    # Clean 'Redistribution Value' and convert to float
    df['Redistribution Value'] = (
        df['Redistribution Value']
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    # Convert 'Delivered_date' to datetime, handling errors
    df['Delivered_date'] = pd.to_datetime(
        df['Delivered_date'], errors='coerce', dayfirst=True
    )
    # Extract 'Month' as Period for consistent monthly grouping
    df['Month'] = df['Delivered_date'].dt.to_period('M')
    # Fill NaN in 'Delivered Qty' with 0
    df['Delivered Qty'] = df['Delivered Qty'].fillna(0)
    # Calculate 'Total_Amount_Spent'
    df['Total_Amount_Spent'] = df['Redistribution Value'] * df['Delivered Qty']

    # Ensure 'Order_Id' is present, generating a unique one if missing
    if 'Order_Id' not in df.columns:
        df['Order_Id'] = df['Customer_Phone'].astype(str) + '_' + \
                         df['Delivered_date'].dt.strftime('%Y%m%d%H%M%S') + '_' + \
                         df.groupby(['Customer_Phone', 'Delivered_date']).cumcount().astype(str)
    return df

def load_model_preds():
    """
    Loads model prediction data, renames columns, and performs type conversions.
    Applies a 'Suggestion' based on 'Probability'.
    """
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

# Load Data (once globally when the app starts)
DF = load_sales_data()
PRED_DF = load_model_preds()

# --- Heuristic Profiling Functions (as provided, with minor adjustments for robustness) ---
def analyze_customer_purchases_extended(df, customer_phone):
    """
    Analyzes a specific customer's purchase history, providing summaries
    at brand and brand-SKU levels.
    """
    customer_df = df[df['Customer_Phone'] == customer_phone].copy()

    if customer_df.empty:
        return f"No data found for customer phone: {customer_phone}"

    customer_df.sort_values('Delivered_date', inplace=True)
    customer_df['Month'] = customer_df['Delivered_date'].dt.to_period('M')

    customer_name = customer_df['Customer_Name'].iloc[0] if not customer_df.empty else 'N/A'
    brands_bought = customer_df['Brand'].unique().tolist()
    total_brands_bought = len(brands_bought)
    brand_skus = customer_df.groupby('Brand')['SKU_Code'].unique().apply(list).to_dict()
    total_unique_skus_bought = customer_df['SKU_Code'].nunique()
    skus_bought = customer_df['SKU_Code'].unique().tolist()

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
    }

    return report

def predict_next_purchases(df_full, customer_phone):
    """
    Predicts next purchases for a given customer based on historical data.
    """
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
    ).head(3)

    overall_next_brand_prediction = customer_df_sorted['Brand'].iloc[-1] if not customer_df_sorted.empty else 'N/A'

    return {
        'sku_predictions': sku_predictions_df,
        'overall_next_brand_prediction': overall_next_brand_prediction,
    }

# --- Utility function ---
def calculate_brand_pairs(df):
    """Calculates co-occurrence counts for brand pairs within the same order."""
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

# --- Recommender System Data Preparation (Cached) ---
def prepare_recommender_data(df_full):
    """
    Prepares data for the recommender system, including user-item matrix,
    item similarity, and content-based similarity.
    """
    user_item_matrix = df_full.pivot_table(index='Customer_Phone', columns='SKU_Code',
                                           values='Redistribution Value', aggfunc='sum', fill_value=0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity,
                                      index=user_item_matrix.columns,
                                      columns=user_item_matrix.columns)

    item_attributes_cols = ['SKU_Code', 'Brand']
    if 'Branch' in df_full.columns:
        item_attributes_cols.append('Branch')

    item_attributes = df_full[item_attributes_cols].drop_duplicates(subset=['SKU_Code']).set_index('SKU_Code')

    for col in ['Brand', 'Branch']:
        if col in item_attributes.columns:
            item_attributes[col] = item_attributes[col].astype(str).fillna('Unknown')

    encoder = OneHotEncoder(handle_unknown='ignore')
    item_features_encoded = encoder.fit_transform(item_attributes)
    content_similarity = cosine_similarity(item_features_encoded)
    content_similarity_df = pd.DataFrame(content_similarity,
                                          index=item_attributes.index,
                                          columns=item_attributes.index)

    common_skus = item_similarity_df.index.intersection(content_similarity_df.index)
    if common_skus.empty:
        return None, None, None

    filtered_item_similarity = item_similarity_df.loc[common_skus, common_skus]
    filtered_content_similarity = content_similarity_df.loc[common_skus, common_skus]
    hybrid_similarity = (filtered_item_similarity + filtered_content_similarity) / 2

    sku_brand_map = df_full[['SKU_Code', 'Brand']].drop_duplicates(subset='SKU_Code').set_index('SKU_Code')

    return user_item_matrix, hybrid_similarity, sku_brand_map

def recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, sku_brand_map, top_n=5):
    """Recommends SKUs based on a hybrid similarity approach."""
    if customer_phone not in user_item_matrix.index:
        return pd.DataFrame()

    purchased_skus = user_item_matrix.loc[customer_phone]
    purchased_skus = purchased_skus[purchased_skus > 0].index.tolist()

    if not purchased_skus:
        return pd.DataFrame()

    valid_purchased_skus = [sku for sku in purchased_skus if sku in hybrid_similarity.columns]
    if not valid_purchased_skus:
        return pd.DataFrame()

    sku_scores = hybrid_similarity[valid_purchased_skus].mean(axis=1)
    sku_scores = sku_scores.drop(index=[s for s in purchased_skus if s in sku_scores.index], errors='ignore')

    if sku_scores.empty:
        return pd.DataFrame()

    top_skus = sku_scores.sort_values(ascending=False).head(top_n)

    recommendations = sku_brand_map.loc[top_skus.index.intersection(sku_brand_map.index)].copy()
    recommendations['Similarity_Score'] = top_skus.loc[recommendations.index].values
    return recommendations.reset_index()

def combined_report_recommender(customer_phone, user_item_matrix, hybrid_similarity, df_full, sku_brand_map, top_n=5):
    """Generates past purchases and recommendations for a customer."""
    past_purchases = df_full[df_full['Customer_Phone'] == customer_phone][['SKU_Code', 'Brand']].drop_duplicates()
    recommendations = recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, sku_brand_map, top_n)
    return past_purchases, recommendations

# --- Dash App Setup ---
app = dash.Dash(__name__)

# Helper to encode image to base64 for embedding
def b64_image(image_path):
    """Encodes an image to base64 for embedding in HTML."""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('ascii')
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Skipping logo.")
        return "" # Return empty string if logo not found

# Load logo (assuming logo.png is in the same directory)
# Uncomment the line below if you have a 'logo.png' file in your project directory
# logo_base64 = b64_image("logo.png")

app.layout = html.Div([
    # Top Header
    html.Div(
        className="header",
        children=[
            # html.Img(src=logo_base64, style={'height': '50px', 'marginRight': '15px'}), # Uncomment for logo
            html.H1("📊 Sales Intelligence Dashboard", style={'color': '#333', 'textAlign': 'center', 'flexGrow': 1}),
        ],
        style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'padding': '20px',
            'borderBottom': '1px solid #eee',
            'backgroundColor': '#f8f8f8',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
    ),

    # Main Content Area
    html.Div(
        className="main-content",
        style={'display': 'flex', 'flexDirection': 'row', 'padding': '20px', 'minHeight': '80vh'},
        children=[
            # Sidebar
            html.Div(
                className="sidebar",
                style={
                    'width': '20%',
                    'padding': '20px',
                    'borderRight': '1px solid #eee',
                    'backgroundColor': '#f0f2f6',
                    'borderRadius': '8px',
                    'boxShadow': '2px 2px 8px rgba(0,0,0,0.1)',
                    'flexShrink': 0 # Prevent sidebar from shrinking
                },
                children=[
                    html.H2("SALES DASHBOARD 🚀", style={'color': '#555', 'marginBottom': '20px'}),
                    dcc.RadioItems(
                        id='section-selector',
                        options=[
                            {'label': '📊 EDA Overview', 'value': 'eda_overview'},
                            {'label': '📉 Drop Detection', 'value': 'drop_detection'},
                            {'label': '👤 Customer Profiling', 'value': 'customer_profiling'},
                            {'label': '🧑‍💻 Customer Profiling (Model Predictions)', 'value': 'customer_model_predictions'},
                            {'label': '🔁 Cross-Selling', 'value': 'cross_selling'},
                            {'label': '🔗 Brand Correlation', 'value': 'brand_correlation'},
                            {'label': '🤖 Recommender', 'value': 'recommender'}
                        ],
                        value='eda_overview', # Default selected section
                        labelStyle={'display': 'block', 'padding': '10px 0', 'fontWeight': 'bold', 'cursor': 'pointer'},
                        style={'color': '#333'}
                    )
                ]
            ),

            # Content Display Area
            html.Div(
                id='page-content',
                style={'width': '80%', 'paddingLeft': '30px', 'flexGrow': 1}
            )
        ]
    ),

    # Hidden Div to store drill-down state (e.g., selected brand for deep dive)
    dcc.Store(id='drilldown-state', data={
        'eda_overview_tab': 'brands_overview', # Keep track of active tab in EDA
        'selected_brand_eda': None, # Store selected brand for deep dive
        'selected_sku_eda': None # Store selected SKU for deep dive
    })
])

# --- Callbacks for Section Navigation ---
@app.callback(
    Output('page-content', 'children'),
    Input('section-selector', 'value')
)
def display_section(selected_section):
    """Renders the content for the selected main section."""
    # Wrap content in dcc.Loading for better UX
    return dcc.Loading(
        id="loading-main-content",
        type="circle",
        children=html.Div([
            html.H2("Loading...", style={'textAlign': 'center', 'marginTop': '50px'})
        ]),
        style={'position': 'absolute', 'top': '50%', 'left': '50%', 'transform': 'translate(-50%, -50%)'}
    )

@app.callback(
    Output('loading-main-content', 'children'), # Update the content of the loading component
    Output('drilldown-state', 'data', allow_duplicate=True), # Reset drilldown state when switching main sections
    Input('section-selector', 'value'),
    State('drilldown-state', 'data'),
    prevent_initial_call=True
)
def update_main_section_content(selected_section, current_drilldown_state):
    """
    Updates the content of the main display area based on sidebar selection.
    Resets drilldown state when switching sections.
    """
    drilldown_state = current_drilldown_state if current_drilldown_state else {
        'eda_overview_tab': 'brands_overview',
        'selected_brand_eda': None,
        'selected_sku_eda': None
    }
    # Reset drilldown state to default when changing main sections
    drilldown_state['eda_overview_tab'] = 'brands_overview'
    drilldown_state['selected_brand_eda'] = None
    drilldown_state['selected_sku_eda'] = None

    if selected_section == 'eda_overview':
        return html.Div([
            html.H2("Exploratory Data Analysis", style={'marginBottom': '20px'}),
            dcc.Tabs(id='eda-tabs', value='brands_overview', children=[
                dcc.Tab(label='Brands Overview', value='brands_overview', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='SKUs Overview', value='skus_overview', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Brand Deep Dive by SKU', value='brand_deep_dive', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='SKU Deep Dive (Monthly Trend)', value='sku_deep_dive', className='custom-tab', selected_className='custom-tab--selected'), # New tab for SKU drill-down
                dcc.Tab(label='Customers Overview', value='customers_overview', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Overall Trends', value='overall_trends', className='custom-tab', selected_className='custom-tab--selected'),
            ], className='custom-tabs-container'),
            html.Div(id='eda-tab-content', style={'marginTop': '20px'})
        ]), drilldown_state
    elif selected_section == 'drop_detection':
        # Drop Detection logic
        brand_month_revenue = DF.groupby(['Brand', 'Month'])['Redistribution Value'].sum().unstack(fill_value=0)
        mom_change = brand_month_revenue.pct_change(axis=1) * 100

        display_df_data = []
        for i, col in enumerate(brand_month_revenue.columns):
            prev_month_col = brand_month_revenue.columns[i-1] if i > 0 else None
            mom_val = mom_change[col]
            prev_revenue = brand_month_revenue[prev_month_col] if prev_month_col is not None else pd.Series(index=brand_month_revenue.index)

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
                display_df_data.append({
                    'Brand': brand,
                    'Month': str(col), # Convert Period to string for display
                    'MoM Change': f"{mom_str}{arrow}",
                    'Previous Month Revenue': prev_str
                })
        
        display_df = pd.DataFrame(display_df_data)

        negative_mom_changes = mom_change[mom_change < 0].stack().reset_index(name='MoM Change')
        
        if not negative_mom_changes.empty:
            melted_revenue = brand_month_revenue.melt(ignore_index=False, var_name='Month', value_name='Previous Month Revenue').reset_index()
            negative_brands_info = pd.merge(negative_mom_changes, melted_revenue, on=['Brand', 'Month'], how='left')
            negative_brands_info['MoM Change Formatted'] = negative_brands_info['MoM Change'].apply(lambda x: f"{x:.1f}%🔻" if pd.notna(x) else "No Data")
            negative_brands_info['Previous Month Revenue Formatted'] = negative_brands_info['Previous Month Revenue'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "No Data")
            
            display_negative = negative_brands_info[['Brand', 'Month', 'MoM Change Formatted', 'Previous Month Revenue Formatted']]
            display_negative.columns = ['Brand', 'Month', 'MoM Change', 'Previous Month Revenue']

            return html.Div([
                html.H3("Brand-Level Month-over-Month (MoM) Revenue Drop Analysis"),
                html.P("This section analyzes the month-over-month percentage change in revenue for each brand to identify significant drops."),
                html.P("NB: Values in the table represent the MoM percentage change in revenue. Upward trend is indicated by ⬆️, and downward trend by🔻. Previous month's revenue is shown in parentheses to provide context."),
                dcc.Graph(
                    figure=px.bar(
                        display_df,
                        x='Month',
                        y='MoM Change',
                        color='Brand',
                        title='MoM Revenue Change by Brand',
                        hover_data=['Previous Month Revenue']
                    ).update_layout(hovermode="x unified") # Improve hover experience
                ),
                html.H3("Brands with Negative Month-over-Month Revenue Change"),
                html.P("The following table highlights brands that experienced a decrease in revenue compared to the previous month."),
                dash.dash_table.DataTable(
                    id='negative-mom-table',
                    columns=[{"name": i, "id": i} for i in display_negative.columns],
                    data=display_negative.to_dict('records'),
                    style_table={'overflowX': 'auto', 'minWidth': '100%'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                )
            ]), drilldown_state
        else:
            return html.Div([
                html.H3("Brand-Level Month-over-Month (MoM) Revenue Drop Analysis"),
                html.P("No brands experienced a month-over-month revenue decrease in the selected period.")
            ]), drilldown_state

    elif selected_section == 'customer_profiling':
        customer_options = [{'label': f"{c} ({DF[DF['Customer_Phone'] == c]['Customer_Name'].iloc[0]})", 'value': c}
                            for c in sorted(DF['Customer_Phone'].unique())]
        return html.Div([
            html.H3("Customer Purchase Deep-Dive"),
            html.P("Select a customer from the dropdown to view their detailed purchase history and insights."),
            dcc.Dropdown(
                id='customer-profiling-dropdown',
                options=customer_options,
                placeholder="Select Customer Phone",
                value=None,
                clearable=True,
                style={'marginBottom': '20px'}
            ),
            html.Div(id='customer-profiling-output', style={'marginTop': '20px'})
        ]), drilldown_state
    elif selected_section == 'customer_model_predictions':
        customer_options = [{'label': f"{c}", 'value': c} for c in sorted(PRED_DF['Customer_Phone'].unique())]
        return html.Div([
            html.H3("Next-Purchase Model Predictions"),
            html.P("View AI-powered predictions for a customer's next brand purchase, date, and expected spend/quantity."),
            dcc.Dropdown(
                id='model-predictions-dropdown',
                options=customer_options,
                placeholder="Select Customer Phone",
                value=None,
                clearable=True,
                style={'marginBottom': '20px'}
            ),
            html.Div(id='model-predictions-output', style={'marginTop': '20px'})
        ]), drilldown_state
    elif selected_section == 'cross_selling':
        lp = DF.groupby(['Customer_Phone','Brand'])['Month'].max().reset_index()
        drop = lp[lp['Month'] < lp['Month'].max()]
        sw = DF.merge(drop, on='Customer_Phone', suffixes=('','_dropped'))
        sw = sw[(sw['Month'] > sw['Month_dropped']) & (sw['Brand'] != sw['Brand_dropped'])]
        patterns = sw.groupby(['Brand_dropped','Brand']).size().reset_index(name='Count')
        top3 = patterns.sort_values(['Brand_dropped','Count'], ascending=[True,False]).groupby('Brand_dropped').head(3)
        
        if not top3.empty:
            return html.Div([
                html.H3("Brand Switching Patterns (Top 3)"),
                html.P("This table shows the top 3 brands customers switch to from a previously purchased brand."),
                dash.dash_table.DataTable(
                    id='brand-switching-table',
                    columns=[{"name": i, "id": i} for i in top3.columns],
                    data=top3.to_dict('records'),
                    style_table={'overflowX': 'auto', 'minWidth': '100%'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                )
            ]), drilldown_state
        else:
            return html.Div([
                html.H3("Brand Switching Patterns (Top 3)"),
                html.P("No brand switching patterns found in the data.")
            ]), drilldown_state
    elif selected_section == 'brand_correlation':
        mat = DF.groupby(['Customer_Phone','Brand'])['Order_Id'].count().unstack(fill_value=0)
        corr_matrix = mat.corr().round(2).reset_index()
        
        if not corr_matrix.empty:
            return html.Div([
                html.H3("Brand Correlation Matrix"),
                html.P("This matrix shows the correlation between different brands based on customer purchase behavior. A higher value indicates brands are frequently bought together."),
                dash.dash_table.DataTable(
                    id='brand-correlation-table',
                    columns=[{"name": i, "id": i} for i in corr_matrix.columns],
                    data=corr_matrix.to_dict('records'),
                    style_table={'overflowX': 'auto', 'minWidth': '100%'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'center', 'padding': '10px'}
                )
            ]), drilldown_state
        else:
            return html.Div([
                html.H3("Brand Correlation Matrix"),
                html.P("Cannot generate brand correlation matrix. Insufficient data.")
            ]), drilldown_state
    elif selected_section == 'recommender':
        user_item_matrix, hybrid_similarity, sku_brand_map = prepare_recommender_data(DF)
        if user_item_matrix is None or hybrid_similarity is None or sku_brand_map is None:
            recommender_content = html.Div([
                html.H3("Hybrid SKU Recommendations"),
                html.P("Recommender system could not be initialized due to missing data or common SKUs. Ensure your dataset has enough variety for recommendations.")
            ])
        else:
            customer_list = sorted(user_item_matrix.index.tolist())
            recommender_content = html.Div([
                html.H3("Hybrid SKU Recommendations"),
                html.P("This section provides personalized SKU recommendations based on a hybrid approach, combining your past purchases with similar customers' behavior and item attributes."),
                dcc.Dropdown(
                    id='recommender-customer-dropdown',
                    options=[{'label': c, 'value': c} for c in customer_list],
                    placeholder="Select Customer for Recommendations:",
                    value=None,
                    clearable=True,
                    style={'marginBottom': '10px'}
                ),
                html.Button('Generate Recommendations', id='generate-recommendations-button', n_clicks=0,
                            style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'padding': '10px 15px', 'borderRadius': '5px', 'cursor': 'pointer', 'marginBottom': '20px'}),
                html.Div(id='recommender-output')
            ])
        return recommender_content, drilldown_state
    return html.Div("Select a section from the sidebar."), drilldown_state

# --- Callbacks for EDA Tabs and Drill-Down Logic ---
@app.callback(
    Output('eda-tab-content', 'children'),
    Output('drilldown-state', 'data', allow_duplicate=True),
    Input('eda-tabs', 'value'),
    Input('brands-overview-chart', 'clickData'), # Input for drill-down from Brands Overview
    Input('brand-deep-dive-back-button', 'n_clicks'), # Back button for brand deep dive
    Input('sku-deep-dive-back-button', 'n_clicks'), # Back button for SKU deep dive
    Input('brand-deep-dive-skus-rev-chart', 'clickData'), # Input for drill-down from Brand Deep Dive to SKU
    State('drilldown-state', 'data'), # Get current drilldown state
    prevent_initial_call=True
)
def render_eda_tab_content(tab_value, click_data_brands_overview, back_button_clicks_brand,
                           back_button_clicks_sku, click_data_brand_skus_rev, current_drilldown_state):
    """
    Manages content for EDA tabs and handles drill-down/drill-up navigation.
    """
    ctx = dash.callback_context

    # Initialize drilldown state if not present
    drilldown_state = current_drilldown_state if current_drilldown_state else {
        'eda_overview_tab': 'brands_overview',
        'selected_brand_eda': None,
        'selected_sku_eda': None
    }

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'no_trigger'
    
    # --- Handle Drill-Up (Back Buttons) ---
    if triggered_id == 'sku-deep-dive-back-button' and back_button_clicks_sku and back_button_clicks_sku > 0:
        # From SKU Deep Dive back to Brand Deep Dive
        drilldown_state['selected_sku_eda'] = None
        tab_value = 'brand_deep_dive'
        drilldown_state['eda_overview_tab'] = 'brand_deep_dive'
    elif triggered_id == 'brand-deep-dive-back-button' and back_button_clicks_brand and back_button_clicks_brand > 0:
        # From Brand Deep Dive back to Brands Overview
        drilldown_state['selected_brand_eda'] = None
        drilldown_state['selected_sku_eda'] = None # Ensure SKU state is also cleared
        tab_value = 'brands_overview'
        drilldown_state['eda_overview_tab'] = 'brands_overview'
    
    # --- Handle Drill-Down ---
    elif triggered_id == 'brands-overview-chart' and click_data_brands_overview:
        # Drill down from Brands Overview to Brand Deep Dive
        clicked_brand = click_data_brands_overview['points'][0]['x']
        drilldown_state['selected_brand_eda'] = clicked_brand
        drilldown_state['selected_sku_eda'] = None # Clear SKU if drilling down from brand
        tab_value = 'brand_deep_dive'
        drilldown_state['eda_overview_tab'] = 'brand_deep_dive'
    elif triggered_id == 'brand-deep-dive-skus-rev-chart' and click_data_brand_skus_rev:
        # Drill down from Brand Deep Dive to SKU Deep Dive
        clicked_sku = click_data_brand_skus_rev['points'][0]['x']
        drilldown_state['selected_sku_eda'] = clicked_sku
        tab_value = 'sku_deep_dive'
        drilldown_state['eda_overview_tab'] = 'sku_deep_dive'
    else:
        # Regular tab selection or initial load
        # If switching tabs, ensure drill-down states are consistent
        if tab_value != 'brand_deep_dive' and tab_value != 'sku_deep_dive':
            drilldown_state['selected_brand_eda'] = None
            drilldown_state['selected_sku_eda'] = None
        drilldown_state['eda_overview_tab'] = tab_value # Update active tab in state


    content = html.Div() # Default empty content

    if tab_value == 'brands_overview':
        data_rev_brand = DF.groupby("Brand")["Redistribution Value"].sum().nlargest(10).reset_index()
        fig_rev_brand = px.bar(data_rev_brand, x='Brand', y='Redistribution Value',
                               title='Top 10 Brands by Total Revenue (Click to Deep Dive)',
                               labels={'Redistribution Value': 'Total Revenue'},
                               hover_data={'Redistribution Value': ':.2s'})
        fig_rev_brand.update_layout(clickmode='event+select') # Enable click events

        data_qty_brand = DF.groupby("Brand")["Delivered Qty"].sum().nlargest(10).reset_index()
        fig_qty_brand = px.bar(data_qty_brand, x='Brand', y='Delivered Qty',
                               title='Top 10 Brands by Quantity Sold',
                               labels={'Delivered Qty': 'Quantity Sold'})

        share_brand = (DF.groupby("Brand")["Delivered Qty"].sum() / DF["Delivered Qty"].sum() * 100).nlargest(10).reset_index()
        fig_share_brand = px.bar(share_brand, x='Brand', y='Delivered Qty',
                                 title='Brands by Share of Total Quantity (Top 10)',
                                 labels={'Delivered Qty': 'Share (%)'})

        df_brand_trend = DF.copy()
        df_brand_trend["MonthTS"] = df_brand_trend["Month"].dt.to_timestamp()
        top10_brands = df_brand_trend.groupby("Brand")["Delivered Qty"].sum().nlargest(10).index
        trend_brand = df_brand_trend[df_brand_trend["Brand"].isin(top10_brands)]
        trend_brand = trend_brand.groupby(["MonthTS", "Brand"])["Delivered Qty"].sum().unstack().reset_index()
        fig_trend_brand = px.line(trend_brand, x='MonthTS', y=trend_brand.columns[1:],
                                  title='Monthly Quantity Trend for Top Brands',
                                  labels={'MonthTS': 'Month', 'value': 'Quantity Sold'})

        df_brand_pairs = calculate_brand_pairs(DF).head(5)
        fig_brand_pairs = px.bar(df_brand_pairs, x='Brand_Pair_Formatted', y='Count',
                                 title='Brand Pair Analysis (Top 5)',
                                 labels={'Brand_Pair_Formatted': 'Brand Pair', 'Count': 'Co-occurrence Count'})

        brand_var = DF.groupby("Customer_Phone")["Brand"].nunique().reset_index()
        dist_brand_var = brand_var['Brand'].value_counts().sort_index().reset_index()
        dist_brand_var.columns = ['Unique Brands Purchased', 'Number of Customers']
        fig_dist_brand_var = px.bar(dist_brand_var, x='Unique Brands Purchased', y='Number of Customers',
                                    title="Distribution of Brand Variety",
                                    labels={'Unique Brands Purchased': 'Unique Brands Purchased', 'Number of Customers': 'Number of Customers'})


        content = html.Div([
            html.H3("Brands Analysis"),
            html.P("Click on a bar in 'Top 10 Brands by Total Revenue' to drill down into Brand Deep Dive.", style={'fontStyle': 'italic', 'color': '#666'}),
            dcc.Graph(id='brands-overview-chart', figure=fig_rev_brand),
            dcc.Graph(figure=fig_qty_brand),
            dcc.Graph(figure=fig_share_brand),
            dcc.Graph(figure=fig_trend_brand),
            dcc.Graph(figure=fig_brand_pairs),
            dcc.Graph(figure=fig_dist_brand_var)
        ])

    elif tab_value == 'skus_overview':
        data_rev_sku = DF.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10).reset_index()
        fig_rev_sku = px.bar(data_rev_sku, x='SKU_Code', y='Redistribution Value',
                             title='Top 10 SKUs by Total Revenue',
                             labels={'Redistribution Value': 'Total Revenue'})

        data_qty_sku = DF.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10).reset_index()
        fig_qty_sku = px.bar(data_qty_sku, x='SKU_Code', y='Delivered Qty',
                             title='Top 10 SKUs by Quantity Sold',
                             labels={'Delivered Qty': 'Quantity Sold'})

        share_sku = (DF.groupby("SKU_Code")["Delivered Qty"].sum() / DF["Delivered Qty"].sum() * 100).nlargest(10).reset_index()
        fig_share_sku = px.bar(share_sku, x='SKU_Code', y='Delivered Qty',
                               title='SKUs by Share of Total Quantity (Top 10)',
                               labels={'Delivered Qty': 'Share (%)'})

        df_sku_trend = DF.copy()
        df_sku_trend["MonthTS"] = df_sku_trend["Month"].dt.to_timestamp()
        top10_skus = df_sku_trend.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10).index
        trend_sku = df_sku_trend[df_sku_trend["SKU_Code"].isin(top10_skus)]
        trend_sku = trend_sku.groupby(["MonthTS", "SKU_Code"])["Delivered Qty"].sum().unstack().reset_index()
        fig_trend_sku = px.line(trend_sku, x='MonthTS', y=trend_sku.columns[1:],
                                title='Monthly Quantity Trend for Top SKUs',
                                labels={'MonthTS': 'Month', 'value': 'Quantity Sold'})

        cnt_sku_pairs = Counter()
        df_sku_pairs = DF.copy()
        df_sku_pairs["Order_ID"] = df_sku_pairs["Customer_Phone"].astype(str) + "_" + df_sku_pairs["Delivered_date"].astype(str)
        for s in df_sku_pairs.groupby("Order_ID")["SKU_Code"].apply(set):
            if len(s) > 1:
                for pair in combinations(sorted(s), 2): cnt_sku_pairs[pair] += 1
        df_sku_pairs_top10 = pd.Series(cnt_sku_pairs).nlargest(10).to_frame(name="Count").reset_index()
        df_sku_pairs_top10.columns = ['SKU_Pair_Tuple', 'Count']
        df_sku_pairs_top10['SKU_Pair_Formatted'] = df_sku_pairs_top10['SKU_Pair_Tuple'].apply(lambda t: f"{t[0]} & {t[1]}")
        fig_sku_pairs = px.bar(df_sku_pairs_top10, x='SKU_Pair_Formatted', y='Count',
                               title='Top 10 Most Frequently Bought SKU Pairs',
                               labels={'SKU_Pair_Formatted': 'SKU Pair', 'Count': 'Co-occurrence Count'})

        sku_var = DF.groupby("Customer_Phone")["SKU_Code"].nunique().reset_index()
        dist_sku_var = sku_var['SKU_Code'].value_counts().sort_index().reset_index()
        dist_sku_var.columns = ['Unique SKUs Purchased', 'Number of Customers']
        fig_dist_sku_var = px.bar(dist_sku_var, x='Unique SKUs Purchased', y='Number of Customers',
                                  title="Distribution of SKU Variety",
                                  labels={'Unique SKUs Purchased': 'Unique SKUs Purchased', 'Number of Customers': 'Number of Customers'})

        content = html.Div([
            html.H3("SKUs Analysis"),
            dcc.Graph(figure=fig_rev_sku),
            dcc.Graph(figure=fig_qty_sku),
            dcc.Graph(figure=fig_share_sku),
            dcc.Graph(figure=fig_trend_sku),
            dcc.Graph(figure=fig_sku_pairs),
            dcc.Graph(figure=fig_dist_sku_var)
        ])

    elif tab_value == 'brand_deep_dive':
        selected_brand_deep_dive = drilldown_state.get('selected_brand_eda')
        if selected_brand_deep_dive:
            brand_df = DF[DF['Brand'] == selected_brand_deep_dive]

            df_pairs = calculate_brand_pairs(DF)
            filtered_pairs = df_pairs[
                df_pairs['Brand_Pair_Tuple'].apply(lambda x: selected_brand_deep_dive in x)
            ].sort_values(by='Count', ascending=False)

            fig_co_purchases = px.bar(filtered_pairs, x='Brand_Pair_Formatted', y='Count',
                                      title=f"Brands Purchased Alongside {selected_brand_deep_dive}",
                                      labels={'Brand_Pair_Formatted': 'Co-Purchased Brand', 'Count': 'Frequency'})

            top_skus_rev = brand_df.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10).reset_index()
            fig_top_skus_rev = px.bar(top_skus_rev, x='SKU_Code', y='Redistribution Value',
                                      title=f"Top 10 SKUs by Revenue in {selected_brand_deep_dive} (Click to Deep Dive into SKU)",
                                      labels={'Redistribution Value': 'Total Revenue'})
            fig_top_skus_rev.update_layout(clickmode='event+select') # Enable click events

            top_skus_qty = brand_df.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10).reset_index()
            fig_top_skus_qty = px.bar(top_skus_qty, x='SKU_Code', y='Delivered Qty',
                                      title=f"Top 10 SKUs by Quantity Sold in {selected_brand_deep_dive}",
                                      labels={'Delivered Qty': 'Quantity Sold'})

            df_sku_trend_brand = brand_df.copy()
            df_sku_trend_brand["MonthTS"] = df_sku_trend_brand["Month"].dt.to_timestamp()
            top5_skus_in_brand = df_sku_trend_brand.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(5).index
            trend_sku_brand = df_sku_trend_brand[df_sku_trend_brand["SKU_Code"].isin(top5_skus_in_brand)]
            trend_sku_brand = trend_sku_brand.groupby(["MonthTS", "SKU_Code"])["Delivered Qty"].sum().unstack().reset_index()
            fig_trend_sku_brand = px.line(trend_sku_brand, x='MonthTS', y=trend_sku_brand.columns[1:],
                                          title=f"Monthly Quantity Trend for Top 5 SKUs in {selected_brand_deep_dive}",
                                          labels={'MonthTS': 'Month', 'value': 'Quantity Sold'})

            content = html.Div([
                html.Button("← Back to Brands Overview", id='brand-deep-dive-back-button', n_clicks=0,
                            style={'marginBottom': '20px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '10px 15px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.H3(f"Brand Deep Dive: {selected_brand_deep_dive}"),
                html.P("Click on a bar in 'Top 10 SKUs by Revenue' to drill down into a specific SKU's monthly trend.", style={'fontStyle': 'italic', 'color': '#666'}),
                dcc.Graph(figure=fig_co_purchases),
                dcc.Graph(id='brand-deep-dive-skus-rev-chart', figure=fig_top_skus_rev), # Add ID for clickData
                dcc.Graph(figure=fig_top_skus_qty),
                dcc.Graph(figure=fig_trend_sku_brand)
            ])
        else:
            content = html.Div([
                html.H3("Brand Deep Dive by SKU"),
                html.P("Select a brand from the 'Brands Overview' tab or the dropdown below to see a deep dive. (No brand selected or invalid state)"),
                dcc.Dropdown(
                    id='brand-deep-dive-dropdown',
                    options=[{'label': b, 'value': b} for b in sorted(DF['Brand'].unique())],
                    placeholder="Select a Brand for Deeper SKU Analysis:",
                    value=None,
                    clearable=True,
                    style={'marginBottom': '20px'}
                ),
                html.Div(id='brand-deep-dive-content-from-dropdown', style={'marginTop': '20px'})
            ])
    
    elif tab_value == 'sku_deep_dive':
        selected_sku_deep_dive = drilldown_state.get('selected_sku_eda')
        selected_brand_parent = drilldown_state.get('selected_brand_eda') # Keep context of parent brand
        if selected_sku_deep_dive:
            sku_df = DF[DF['SKU_Code'] == selected_sku_deep_dive].copy()
            sku_df["MonthTS"] = sku_df["Month"].dt.to_timestamp()

            fig_sku_monthly_qty = px.line(sku_df.groupby("MonthTS")["Delivered Qty"].sum().reset_index(),
                                          x='MonthTS', y='Delivered Qty',
                                          title=f"Monthly Quantity Trend for SKU: {selected_sku_deep_dive}",
                                          labels={'Delivered Qty': 'Total Quantity', 'MonthTS': 'Month'})

            fig_sku_monthly_rev = px.line(sku_df.groupby("MonthTS")["Redistribution Value"].sum().reset_index(),
                                          x='MonthTS', y='Redistribution Value',
                                          title=f"Monthly Revenue Trend for SKU: {selected_sku_deep_dive}",
                                          labels={'Redistribution Value': 'Total Revenue', 'MonthTS': 'Month'})
            
            # Top customers for this SKU
            top_customers_sku = sku_df.groupby('Customer_Phone')['Redistribution Value'].sum().nlargest(5).reset_index()
            top_customers_sku['Customer_Info'] = top_customers_sku['Customer_Phone'].astype(str) + ' (' + \
                                                  DF[DF['Customer_Phone'].isin(top_customers_sku['Customer_Phone'])]['Customer_Name'].iloc[0] + ')' # Assuming name is consistent
            fig_top_customers_sku = px.bar(top_customers_sku, x='Redistribution Value', y='Customer_Info', orientation='h',
                                            title=f"Top 5 Customers for SKU: {selected_sku_deep_dive}",
                                            labels={'Redistribution Value': 'Total Spend', 'Customer_Info': 'Customer'})
            fig_top_customers_sku.update_layout(yaxis={'categoryorder':'total ascending'})


            content = html.Div([
                html.Button("← Back to Brand Deep Dive", id='sku-deep-dive-back-button', n_clicks=0,
                            style={'marginBottom': '20px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '10px 15px', 'borderRadius': '5px', 'cursor': 'pointer'}),
                html.H3(f"SKU Deep Dive: {selected_sku_deep_dive} (from Brand: {selected_brand_parent})"),
                dcc.Graph(figure=fig_sku_monthly_qty),
                dcc.Graph(figure=fig_sku_monthly_rev),
                dcc.Graph(figure=fig_top_customers_sku)
            ])
        else:
            content = html.Div([
                html.H3("SKU Deep Dive (Monthly Trend)"),
                html.P("Select an SKU from 'Brand Deep Dive' tab or the dropdown below to see its detailed monthly trends and top customers."),
                dcc.Dropdown(
                    id='sku-deep-dive-dropdown',
                    options=[{'label': s, 'value': s} for s in sorted(DF['SKU_Code'].unique())],
                    placeholder="Select an SKU for Deep Dive:",
                    value=None,
                    clearable=True,
                    style={'marginBottom': '20px'}
                ),
                html.Div(id='sku-deep-dive-content-from-dropdown', style={'marginTop': '20px'})
            ])

    elif tab_value == 'customers_overview':
        df_chart_cust_rev = DF.copy()
        df_chart_cust_rev['Customer_Info'] = df_chart_cust_rev['Customer_Name'] + ' (0' + df_chart_cust_rev['Customer_Phone'].astype(str) + ')'
        customer_ltv_with_name = (
            df_chart_cust_rev.groupby("Customer_Info")["Redistribution Value"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_cust_rev = px.bar(customer_ltv_with_name, x='Redistribution Value', y='Customer_Info', orientation='h',
                              title='Top 10 Customers by Total Spending',
                              labels={'Redistribution Value': 'Total Redistribution Value', 'Customer_Info': 'Customer'})
        fig_cust_rev.update_layout(yaxis={'categoryorder':'total ascending'})

        df_chart_cust_qty = DF.copy()
        df_chart_cust_qty['Customer_Info'] = df_chart_cust_qty['Customer_Name'] + ' (0' + df_chart_cust_qty['Customer_Phone'].astype(str) + ')'
        top_buyers_qty_with_name = (
            df_chart_cust_qty.groupby("Customer_Info")["Delivered Qty"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_cust_qty = px.bar(top_buyers_qty_with_name, x='Delivered Qty', y='Customer_Info', orientation='h',
                              title='Top 10 Buyers by Quantity Purchased',
                              labels={'Delivered Qty': 'Total Quantity', 'Customer_Info': 'Customer'})
        fig_cust_qty.update_layout(yaxis={'categoryorder':'total ascending'})

        counts_cust_type = DF.groupby("Customer_Phone")["Delivered_date"].nunique()
        summary_cust_type = (counts_cust_type == 1).map({True: "One-time", False: "Repeat"}).value_counts().reset_index()
        summary_cust_type.columns = ['Buyer Type', 'Count']
        fig_cust_type = px.bar(summary_cust_type, x='Buyer Type', y='Count',
                               title='Buyer Type (Repeat vs. One-Time Buyers)')

        df_cust_trend = DF.copy()
        df_cust_trend["MonthTS"] = df_cust_trend["Month"].dt.to_timestamp()
        top5_cust = df_cust_trend.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(5).index
        trend_cust = df_cust_trend[df_cust_trend["Customer_Phone"].isin(top5_cust)]
        trend_cust = trend_cust.groupby(["MonthTS", "Customer_Phone"])["Redistribution Value"].sum().unstack().reset_index()
        fig_trend_cust = px.line(trend_cust, x='MonthTS', y=trend_cust.columns[1:],
                                 title='Monthly Purchase Value Trend for Top 5 Buyers',
                                 labels={'MonthTS': 'Month', 'value': 'Total Spend'})

        latest_month = DF['Month'].max()
        buyer_performance = DF[DF['Month'] == latest_month].groupby('Customer_Phone')['Redistribution Value'].sum()
        top5_buyers_latest = buyer_performance.nlargest(5).reset_index()
        bottom5_buyers_latest = buyer_performance.nsmallest(5).reset_index()

        fig_top_buyers = px.bar(top5_buyers_latest, x='Customer_Phone', y='Redistribution Value',
                                title='Top 5 Buyers (Latest Month)')
        fig_bottom_buyers = px.bar(bottom5_buyers_latest, x='Customer_Phone', y='Redistribution Value',
                                   title='Bottom 5 Buyers (Latest Month)')

        content = html.Div([
            html.H3("Customers Analysis"),
            dcc.Graph(figure=fig_cust_rev),
            dcc.Graph(figure=fig_cust_qty),
            dcc.Graph(figure=fig_cust_type),
            dcc.Graph(figure=fig_trend_cust),
            html.Div([
                html.Div(dcc.Graph(figure=fig_top_buyers), style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div(dcc.Graph(figure=fig_bottom_buyers), style={'width': '48%', 'display': 'inline-block'})
            ])
        ])

    elif tab_value == 'overall_trends':
        monthly_summary_overall = DF.groupby("Month")[["Delivered Qty", "Redistribution Value"]].sum().reset_index()
        monthly_summary_overall["MonthTS"] = monthly_summary_overall["Month"].dt.to_timestamp()

        fig_overall_qty = px.line(monthly_summary_overall, x='MonthTS', y='Delivered Qty',
                                  title='Overall Monthly Quantity Trend',
                                  labels={'Delivered Qty': 'Total Quantity'})
        fig_overall_rev = px.line(monthly_summary_overall, x='MonthTS', y='Redistribution Value',
                                  title='Overall Monthly Revenue Trend',
                                  labels={'Redistribution Value': 'Total Revenue'})

        avg_order_value = DF.groupby("Month")["Redistribution Value"].mean().reset_index()
        avg_order_value["MonthTS"] = avg_order_value["Month"].dt.to_timestamp()
        fig_aov = px.line(avg_order_value, x='MonthTS', y='Redistribution Value',
                          title='Average Order Value Trend',
                          labels={'Redistribution Value': 'Average Order Value'})

        ltv_top10 = DF.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(10).reset_index()
        ltv_top10.columns = ['Customer Phone', 'Total Spend']
        fig_ltv_top10 = px.bar(ltv_top10, x='Customer Phone', y='Total Spend',
                               title='Top 10 Customers by Lifetime Value (Total Spend)')

        content = html.Div([
            html.H3("Overall Sales Trends"),
            dcc.Graph(figure=fig_overall_qty),
            dcc.Graph(figure=fig_overall_rev),
            dcc.Graph(figure=fig_aov),
            dcc.Graph(figure=fig_ltv_top10)
        ])
    
    # If the tab is 'brand_deep_dive' but no brand is selected (e.g., direct access or initial load)
    # and the trigger wasn't a drill-down, render the dropdown.
    if tab_value == 'brand_deep_dive' and not drilldown_state.get('selected_brand_eda') and triggered_id not in ['brands-overview-chart']:
        content = html.Div([
            html.H3("Brand Deep Dive by SKU"),
            html.P("Select a brand from the 'Brands Overview' tab or the dropdown below to see a deep dive."),
            dcc.Dropdown(
                id='brand-deep-dive-dropdown',
                options=[{'label': b, 'value': b} for b in sorted(DF['Brand'].unique())],
                placeholder="Select a Brand for Deeper SKU Analysis:",
                value=None,
                clearable=True,
                style={'marginBottom': '20px'}
            ),
            html.Div(id='brand-deep-dive-content-from-dropdown', style={'marginTop': '20px'})
        ])
    
    # If the tab is 'sku_deep_dive' but no SKU is selected
    if tab_value == 'sku_deep_dive' and not drilldown_state.get('selected_sku_eda') and triggered_id not in ['brand-deep-dive-skus-rev-chart']:
        content = html.Div([
            html.H3("SKU Deep Dive (Monthly Trend)"),
            html.P("Select an SKU from 'Brand Deep Dive' tab or the dropdown below to see its detailed monthly trends and top customers."),
            dcc.Dropdown(
                id='sku-deep-dive-dropdown',
                options=[{'label': s, 'value': s} for s in sorted(DF['SKU_Code'].unique())],
                placeholder="Select an SKU for Deep Dive:",
                value=None,
                clearable=True,
                style={'marginBottom': '20px'}
            ),
            html.Div(id='sku-deep-dive-content-from-dropdown', style={'marginTop': '20px'})
        ])


    return content, drilldown_state

# Callback for Brand Deep Dive dropdown (if not drilled down from chart)
@app.callback(
    Output('brand-deep-dive-content-from-dropdown', 'children'),
    Input('brand-deep-dive-dropdown', 'value'),
    prevent_initial_call=True
)
def update_brand_deep_dive_from_dropdown(selected_brand_deep_dive):
    """Updates Brand Deep Dive content when a brand is selected from the dropdown."""
    if selected_brand_deep_dive:
        brand_df = DF[DF['Brand'] == selected_brand_deep_dive]

        df_pairs = calculate_brand_pairs(DF)
        filtered_pairs = df_pairs[
            df_pairs['Brand_Pair_Tuple'].apply(lambda x: selected_brand_deep_dive in x)
        ].sort_values(by='Count', ascending=False)

        fig_co_purchases = px.bar(filtered_pairs, x='Brand_Pair_Formatted', y='Count',
                                  title=f"Brands Purchased Alongside {selected_brand_deep_dive}",
                                  labels={'Brand_Pair_Formatted': 'Co-Purchased Brand', 'Count': 'Frequency'})

        top_skus_rev = brand_df.groupby("SKU_Code")["Redistribution Value"].sum().nlargest(10).reset_index()
        fig_top_skus_rev = px.bar(top_skus_rev, x='SKU_Code', y='Redistribution Value',
                                  title=f"Top 10 SKUs by Revenue in {selected_brand_deep_dive} (Click to Deep Dive into SKU)",
                                  labels={'Redistribution Value': 'Total Revenue'})
        fig_top_skus_rev.update_layout(clickmode='event+select') # Enable click events

        top_skus_qty = brand_df.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(10).reset_index()
        fig_top_skus_qty = px.bar(top_skus_qty, x='SKU_Code', y='Delivered Qty',
                                  title=f"Top 10 SKUs by Quantity Sold in {selected_brand_deep_dive}",
                                  labels={'Delivered Qty': 'Quantity Sold'})

        df_sku_trend_brand = brand_df.copy()
        df_sku_trend_brand["MonthTS"] = df_sku_trend_brand["Month"].dt.to_timestamp()
        top5_skus_in_brand = df_sku_trend_brand.groupby("SKU_Code")["Delivered Qty"].sum().nlargest(5).index
        trend_sku_brand = df_sku_trend_brand[df_sku_trend_brand["SKU_Code"].isin(top5_skus_in_brand)]
        trend_sku_brand = trend_sku_brand.groupby(["MonthTS", "SKU_Code"])["Delivered Qty"].sum().unstack().reset_index()
        fig_trend_sku_brand = px.line(trend_sku_brand, x='MonthTS', y=trend_sku_brand.columns[1:],
                                      title=f"Monthly Quantity Trend for Top 5 SKUs in {selected_brand_deep_dive}",
                                      labels={'MonthTS': 'Month', 'value': 'Quantity Sold'})

        return html.Div([
            html.H4(f"Deep Dive Details for {selected_brand_deep_dive}"),
            html.P("Click on a bar in 'Top 10 SKUs by Revenue' to drill down into a specific SKU's monthly trend.", style={'fontStyle': 'italic', 'color': '#666'}),
            dcc.Graph(figure=fig_co_purchases),
            dcc.Graph(id='brand-deep-dive-skus-rev-chart', figure=fig_top_skus_rev), # Add ID for clickData
            dcc.Graph(figure=fig_top_skus_qty),
            dcc.Graph(figure=fig_trend_sku_brand)
        ])
    return html.Div("Please select a brand.")

# Callback for SKU Deep Dive dropdown (if not drilled down from chart)
@app.callback(
    Output('sku-deep-dive-content-from-dropdown', 'children'),
    Input('sku-deep-dive-dropdown', 'value'),
    prevent_initial_call=True
)
def update_sku_deep_dive_from_dropdown(selected_sku_deep_dive):
    """Updates SKU Deep Dive content when an SKU is selected from the dropdown."""
    if selected_sku_deep_dive:
        sku_df = DF[DF['SKU_Code'] == selected_sku_deep_dive].copy()
        sku_df["MonthTS"] = sku_df["Month"].dt.to_timestamp()

        fig_sku_monthly_qty = px.line(sku_df.groupby("MonthTS")["Delivered Qty"].sum().reset_index(),
                                      x='MonthTS', y='Delivered Qty',
                                      title=f"Monthly Quantity Trend for SKU: {selected_sku_deep_dive}",
                                      labels={'Delivered Qty': 'Total Quantity', 'MonthTS': 'Month'})

        fig_sku_monthly_rev = px.line(sku_df.groupby("MonthTS")["Redistribution Value"].sum().reset_index(),
                                      x='MonthTS', y='Redistribution Value',
                                      title=f"Monthly Revenue Trend for SKU: {selected_sku_deep_dive}",
                                      labels={'Redistribution Value': 'Total Revenue', 'MonthTS': 'Month'})
        
        # Top customers for this SKU
        top_customers_sku = sku_df.groupby('Customer_Phone')['Redistribution Value'].sum().nlargest(5).reset_index()
        # Ensure 'Customer_Name' is correctly mapped, assuming Customer_Phone is unique for customer
        customer_names_map = DF[['Customer_Phone', 'Customer_Name']].drop_duplicates().set_index('Customer_Phone')['Customer_Name']
        top_customers_sku['Customer_Info'] = top_customers_sku['Customer_Phone'].astype(str) + ' (' + \
                                              top_customers_sku['Customer_Phone'].map(customer_names_map).fillna('Unknown Name') + ')'
        
        fig_top_customers_sku = px.bar(top_customers_sku, x='Redistribution Value', y='Customer_Info', orientation='h',
                                        title=f"Top 5 Customers for SKU: {selected_sku_deep_dive}",
                                        labels={'Redistribution Value': 'Total Spend', 'Customer_Info': 'Customer'})
        fig_top_customers_sku.update_layout(yaxis={'categoryorder':'total ascending'})


        return html.Div([
            html.H4(f"Deep Dive Details for SKU: {selected_sku_deep_dive}"),
            dcc.Graph(figure=fig_sku_monthly_qty),
            dcc.Graph(figure=fig_sku_monthly_rev),
            dcc.Graph(figure=fig_top_customers_sku)
        ])
    return html.Div("Please select an SKU.")


# --- Callbacks for Customer Profiling ---
@app.callback(
    Output('customer-profiling-output', 'children'),
    Input('customer-profiling-dropdown', 'value')
)
def update_customer_profiling(selected_customer_phone):
    """Displays detailed profiling for a selected customer."""
    if selected_customer_phone:
        report = analyze_customer_purchases_extended(DF, selected_customer_phone)
        heuristic_predictions = predict_next_purchases(DF, selected_customer_phone)

        if isinstance(report, str): # Handle case where customer data is not found
            return html.Div(report, style={'color': 'red', 'fontWeight': 'bold'})
        else:
            brand_summary_df = pd.DataFrame.from_dict(report['Brand Level Summary'], orient='index')
            brand_summary_df = brand_summary_df.rename_axis('Brand').reset_index()

            sku_summary_elements = []
            for brand, sku_summary in report['Brand SKU Level Summary'].items():
                sku_summary_df = pd.DataFrame.from_dict(sku_summary, orient='index')
                sku_summary_df = sku_summary_df.rename_axis('SKU Code').reset_index()
                sku_summary_elements.append(html.H5(f"Brand: {brand}", style={'marginTop': '15px', 'marginBottom': '5px', 'color': '#444'}))
                sku_summary_elements.append(dash.dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in sku_summary_df.columns],
                    data=sku_summary_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'marginBottom': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                    style_header={'backgroundColor': '#e0e0e0', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                ))

            sku_predictions_table = html.Div(html.P("Not enough historical data to provide detailed SKU purchase predictions for this customer.", style={'fontStyle': 'italic', 'color': '#666'}))
            if not heuristic_predictions['sku_predictions'].empty:
                display_sku_preds = heuristic_predictions['sku_predictions'][[
                    'Likely Brand', 'SKU Code', 'Likely Purchase Date', 'Expected Quantity', 'Expected Spend'
                ]]
                sku_predictions_table = dash.dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in display_sku_preds.columns],
                    data=display_sku_preds.to_dict('records'),
                    style_table={'overflowX': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                    style_header={'backgroundColor': '#e0e0e0', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                )

            return html.Div([
                html.P(f"**Customer Name:** {report['Customer Name']}", style={'fontWeight': 'bold'}),
                html.P(f"**Customer Phone:** {report['Customer Phone']}", style={'fontWeight': 'bold'}),
                html.P(f"**Customer Branch:** {report['Customer Branch']}"),
                html.P(f"**Total Unique Brands Bought:** {report['Total Unique Brands Bought']}"),
                html.P(f"**Brands Bought:** {', '.join(report['Brands Bought'])}"),
                html.P(f"**Total Order Count:** {report['Total Order Count']}"),
                html.P(f"**Top Salesperson:** {report['Top Salesperson']}"),
                html.P(f"**Salesperson Designation:** {report['Salesperson Designation']}"),
                html.P(f"**Total Unique SKUs Bought:** {report['Total Unique SKUs Bought']}"),
                html.P(f"**SKUs Bought:** {', '.join(report['SKUs Bought'])}"),

                html.H4("Brand Level Purchase Summary", style={'marginTop': '30px', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                dash.dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in brand_summary_df.columns],
                    data=brand_summary_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                    style_header={'backgroundColor': '#e0e0e0', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                ),

                html.H4("Brand SKU Level Purchase Summary", style={'marginTop': '30px', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                html.Div(sku_summary_elements),

                html.H4("Next Purchase Predictions (Heuristic)", style={'marginTop': '30px', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                html.P(f"**Overall Most Recently Bought Brand (for context):** {heuristic_predictions['overall_next_brand_prediction']}", style={'fontStyle': 'italic'}),
                html.P("**Predicted Next Purchases (SKU Level with Brand and Date/Day):**"),
                sku_predictions_table
            ])
    return html.Div("Please select a customer.", style={'textAlign': 'center', 'color': '#888', 'marginTop': '50px'})

# --- Callbacks for Model Predictions ---
@app.callback(
    Output('model-predictions-output', 'children'),
    Input('model-predictions-dropdown', 'value')
)
def update_model_predictions(selected_customer_phone):
    """Displays model predictions for a selected customer."""
    if selected_customer_phone:
        p = PRED_DF[PRED_DF['Customer_Phone'] == selected_customer_phone].drop(columns=['Customer_Phone']).set_index('SKU_Code')
        p['Probability'] = p['Probability'].map(lambda x: f"{x:.1f}%")
        
        if not p.empty:
            return dash.dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in p.reset_index().columns], # Reset index to make SKU_Code a column
                data=p.reset_index().to_dict('records'),
                style_table={'overflowX': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                style_header={'backgroundColor': '#e0e0e0', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'left', 'padding': '10px'}
            )
        else:
            return html.Div(f"No model predictions available for customer {selected_customer_phone}.", style={'color': '#666', 'fontStyle': 'italic'})
    return html.Div("Please select a customer.", style={'textAlign': 'center', 'color': '#888', 'marginTop': '50px'})

# --- Callback for Recommender System ---
@app.callback(
    Output('recommender-output', 'children'),
    Input('generate-recommendations-button', 'n_clicks'),
    State('recommender-customer-dropdown', 'value'),
    prevent_initial_call=True
)
def generate_recommender_output(n_clicks, sel_customer_recommender):
    """Generates and displays SKU recommendations for a selected customer."""
    if n_clicks > 0 and sel_customer_recommender:
        user_item_matrix, hybrid_similarity, sku_brand_map = prepare_recommender_data(DF)
        if user_item_matrix is None or hybrid_similarity is None or sku_brand_map is None:
            return html.Div("Recommender system could not be initialized due to missing data or common SKUs. Ensure your dataset has enough variety for recommendations.", style={'color': 'red', 'fontWeight': 'bold'})

        past_purchases_df, recommendations_df = combined_report_recommender(
            sel_customer_recommender, user_item_matrix, hybrid_similarity, DF, sku_brand_map, top_n=5
        )

        past_purchases_content = html.Div([
            html.H4("Previously Purchased SKUs", style={'color': '#333'}),
            html.P("No past purchase data found for this customer.", style={'fontStyle': 'italic', 'color': '#666'})
        ])
        if not past_purchases_df.empty:
            past_purchases_content = html.Div([
                html.H4("Previously Purchased SKUs", style={'color': '#333'}),
                dash.dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in past_purchases_df.columns],
                    data=past_purchases_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                    style_header={'backgroundColor': '#e0e0e0', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                )
            ])

        recommendations_content = html.Div([
            html.H4("Recommended SKUs", style={'color': '#333'}),
            html.P("No new recommendations could be generated for this customer.", style={'fontStyle': 'italic', 'color': '#666'})
        ])
        if not recommendations_df.empty:
            recommendations_content = html.Div([
                html.H4("Recommended SKUs", style={'color': '#333'}),
                dash.dash_table.DataTable(
                    columns=[
                        {"name": "SKU Code", "id": "SKU_Code"},
                        {"name": "Brand", "id": "Brand"},
                        {"name": "Similarity Score", "id": "Similarity_Score", "type": "numeric", "format": dash.dash_table.Format.Format(precision=4, scheme=dash.dash_table.Format.Scheme.fixed)}
                    ],
                    data=recommendations_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                    style_header={'backgroundColor': '#e0e0e0', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'left', 'padding': '10px'}
                ),
                html.P("A higher 'Similarity Score' indicates a stronger recommendation.", style={'marginTop': '10px', 'fontStyle': 'italic', 'color': '#666'})
            ])

        return html.Div([
            html.H3(f"Recommendations for Customer {sel_customer_recommender}", style={'marginBottom': '20px'}),
            past_purchases_content,
            html.Hr(style={'margin': '30px 0'}),
            recommendations_content
        ])
    return html.Div("Please select a customer and click 'Generate Recommendations'.", style={'textAlign': 'center', 'color': '#888', 'marginTop': '50px'})


# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True)
