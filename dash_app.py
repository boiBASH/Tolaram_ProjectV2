import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.express as px
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import base64

# --- Data Loading & Preprocessing ---
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
    recommendations = recommend_skus_brands(customer_phone, user_item_matrix, hybrid_similarity, DF, sku_brand_map, top_n)
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
                            {'label': '📈 Hierarchical Analysis', 'value': 'hierarchical_analysis'}, # New top-down section
                            {'label': '📉 Drop Detection', 'value': 'drop_detection'},
                            {'label': '👤 Customer Profiling', 'value': 'customer_profiling'},
                            {'label': '🧑‍💻 Customer Profiling (Model Predictions)', 'value': 'customer_model_predictions'},
                            {'label': '🔁 Cross-Selling', 'value': 'cross_selling'},
                            {'label': '🔗 Brand Correlation', 'value': 'brand_correlation'},
                            {'label': '🤖 Recommender', 'value': 'recommender'}
                        ],
                        value='hierarchical_analysis', # Default selected section
                        labelStyle={'display': 'block', 'padding': '10px 0', 'fontWeight': 'bold', 'cursor': 'pointer'},
                        style={'color': '#333'}
                    )
                ]
            ),

            # Content Display Area
            html.Div(
                id='page-content',
                style={'width': '80%', 'paddingLeft': '30px', 'flexGrow': 1},
                children=[
                    # This dcc.Loading component wraps all content that might be updated by callbacks.
                    # Its children will be rendered while other callbacks are processing.
                    dcc.Loading(
                        id="loading-main-content",
                        type="circle",
                        children=html.Div([
                            html.H2("Loading...", style={'textAlign': 'center', 'marginTop': '50px'})
                        ]),
                        style={'position': 'absolute', 'top': '50%', 'left': '50%', 'transform': 'translate(-50%, -50%)'}
                    ),
                    # ALL hierarchical analysis components must be present in the initial layout
                    # and their visibility controlled by callbacks.
                    html.Div(id='hierarchical-analysis-container', children=[
                        html.Button("← Back to Branches", id='back-to-branches', n_clicks=0, style={'display': 'none'}),
                        html.Button("← Back to Brands in Branch", id='back-to-brands-in-branch', n_clicks=0, style={'display': 'none'}),
                        html.Button("← Back to SKUs in Brand", id='back-to-skus-in-brand', n_clicks=0, style={'display': 'none'}),
                        html.H3(id='hierarchical-title', children="Hierarchical Analysis"),
                        html.P(id='hierarchical-instruction', children="Select a section from the sidebar to begin your analysis.", style={'textAlign': 'center', 'color': '#888', 'marginTop': '50px'}),
                        dcc.Graph(id='branch-overview-chart', figure={}),
                        dcc.Graph(id='branch-brands-chart', figure={}),
                        dcc.Graph(id='branch-brands-chart-qty', figure={}),
                        dcc.Graph(id='brand-skus-chart', figure={}),
                        dcc.Graph(id='brand-skus-chart-qty', figure={}),
                        dcc.Graph(id='customer-details-chart', figure={}),
                        html.H4(id='customer-purchase-details-title', children="Recent Purchase Details for this SKU by Top Customers", style={'display': 'none'}),
                        dash.dash_table.DataTable(id='customer-purchase-details-table', columns=[], data=[])
                    ], style={'display': 'none'}), # Initially hide the entire hierarchical analysis container

                    # Other section containers (initially hidden)
                    html.Div(id='drop-detection-container', style={'display': 'none'}, children=[
                        html.H3("Brand-Level Month-over-Month (MoM) Revenue Drop Analysis"),
                        html.P("This section analyzes the month-over-month percentage change in revenue for each brand to identify significant drops."),
                        html.P("NB: Values in the table represent the MoM percentage change in revenue. Upward trend is indicated by ⬆️, and downward trend by🔻. Previous month's revenue is shown in parentheses to provide context."),
                        dcc.Graph(id='mom-revenue-chart', figure={}),
                        html.H3("Brands with Negative Month-over-Month Revenue Change"),
                        html.P("The following table highlights brands that experienced a decrease in revenue compared to the previous month."),
                        dash.dash_table.DataTable(id='negative-mom-table', columns=[], data=[])
                    ]),
                    html.Div(id='customer-profiling-container', style={'display': 'none'}, children=[
                        html.H3("Customer Purchase Deep-Dive"),
                        html.P("Select a customer from the dropdown to view their detailed purchase history and insights."),
                        dcc.Dropdown(
                            id='customer-profiling-dropdown',
                            options=[],
                            placeholder="Select Customer Phone",
                            value=None,
                            clearable=True,
                            style={'marginBottom': '20px'}
                        ),
                        html.Div(id='customer-profiling-output', style={'marginTop': '20px'})
                    ]),
                    html.Div(id='customer-model-predictions-container', style={'display': 'none'}, children=[
                        html.H3("Next-Purchase Model Predictions"),
                        html.P("View AI-powered predictions for a customer's next brand purchase, date, and expected spend/quantity."),
                        dcc.Dropdown(
                            id='model-predictions-dropdown',
                            options=[],
                            placeholder="Select Customer Phone",
                            value=None,
                            clearable=True,
                            style={'marginBottom': '20px'}
                        ),
                        html.Div(id='model-predictions-output', style={'marginTop': '20px'})
                    ]),
                    html.Div(id='cross-selling-container', style={'display': 'none'}, children=[
                        html.H3("Brand Switching Patterns (Top 3)"),
                        html.P("This table shows the top 3 brands customers switch to from a previously purchased brand."),
                        dash.dash_table.DataTable(id='brand-switching-table', columns=[], data=[])
                    ]),
                    html.Div(id='brand-correlation-container', style={'display': 'none'}, children=[
                        html.H3("Brand Correlation Matrix"),
                        html.P("This matrix shows the correlation between different brands based on customer purchase behavior. A higher value indicates brands are frequently bought together."),
                        dash.dash_table.DataTable(id='brand-correlation-table', columns=[], data=[])
                    ]),
                    html.Div(id='recommender-container', style={'display': 'none'}, children=[
                        html.H3("Hybrid SKU Recommendations"),
                        html.P("This section provides personalized SKU recommendations based on a hybrid approach, combining your past purchases with similar customers' behavior and item attributes."),
                        dcc.Dropdown(
                            id='recommender-customer-dropdown',
                            options=[],
                            placeholder="Select Customer for Recommendations:",
                            value=None,
                            clearable=True,
                            style={'marginBottom': '10px'}
                        ),
                        html.Button('Generate Recommendations', id='generate-recommendations-button', n_clicks=0,
                                    style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'padding': '10px 15px', 'borderRadius': '5px', 'cursor': 'pointer', 'marginBottom': '20px'}),
                        html.Div(id='recommender-output')
                    ])
                ]
            )
        ]
    ),

    # Hidden Div to store drill-down state
    dcc.Store(id='drilldown-state', data={
        'level': 'branch', # 'branch', 'brand', 'sku', 'customer'
        'selected_branch': None,
        'selected_brand': None,
        'selected_sku': None,
        'selected_customer': None
    })
])

# --- Callbacks for Main Section Navigation and Content Display ---
@app.callback(
    # Removed Output('loading-main-content', 'children')
    Output('hierarchical-analysis-container', 'style'),
    Output('drop-detection-container', 'style'),
    Output('customer-profiling-container', 'style'),
    Output('customer-model-predictions-container', 'style'),
    Output('cross-selling-container', 'style'),
    Output('brand-correlation-container', 'style'),
    Output('recommender-container', 'style'),
    Output('drilldown-state', 'data'), # Removed allow_duplicate=True
    Input('section-selector', 'value'),
    State('drilldown-state', 'data'),
    prevent_initial_call=False # Set to False to allow initial rendering of the default section
)
def update_main_section_content(selected_section, current_drilldown_state):
    """
    Updates the content of the main display area based on sidebar selection.
    Controls visibility of main section containers and resets drilldown state.
    """
    # Initialize all containers to hidden
    styles = {
        'hierarchical_analysis': {'display': 'none'},
        'drop_detection': {'display': 'none'},
        'customer_profiling': {'display': 'none'},
        'customer_model_predictions': {'display': 'none'},
        'cross_selling': {'display': 'none'},
        'brand_correlation': {'display': 'none'},
        'recommender': {'display': 'none'},
    }

    # Reset drilldown state to default when changing main sections
    # This ensures that when switching sections, the hierarchical analysis
    # always starts from the top level.
    drilldown_state = {
        'level': 'branch',
        'selected_branch': None,
        'selected_brand': None,
        'selected_sku': None,
        'selected_customer': None
    }

    # Set the selected section to visible
    styles[selected_section] = {'display': 'block'}

    # No content returned for loading-main-content.children, as dcc.Loading manages it.
    return styles['hierarchical_analysis'], styles['drop_detection'], \
           styles['customer_profiling'], styles['customer_model_predictions'], \
           styles['cross_selling'], styles['brand_correlation'], styles['recommender'], \
           drilldown_state

# --- Callbacks for Hierarchical Analysis ---
@app.callback(
    Output('hierarchical-title', 'children'),
    Output('hierarchical-instruction', 'children'),
    Output('branch-overview-chart', 'figure'),
    Output('branch-brands-chart', 'figure'),
    Output('branch-brands-chart-qty', 'figure'),
    Output('brand-skus-chart', 'figure'),
    Output('brand-skus-chart-qty', 'figure'),
    Output('customer-details-chart', 'figure'),
    Output('customer-purchase-details-title', 'style'),
    Output('customer-purchase-details-table', 'columns'),
    Output('customer-purchase-details-table', 'data'),
    Output('back-to-branches', 'style'),
    Output('back-to-brands-in-branch', 'style'),
    Output('back-to-skus-in-brand', 'style'),
    Input('drilldown-state', 'data'),
    prevent_initial_call=True # Set to True to prevent duplicate initial calls
)
def update_hierarchical_analysis_content(drilldown_state):
    """
    Updates the content and visibility of components within the Hierarchical Analysis section
    based on the current drilldown state.
    """
    current_level = drilldown_state['level']
    selected_branch = drilldown_state['selected_branch']
    selected_brand = drilldown_state['selected_brand']
    selected_sku = drilldown_state['selected_sku']
    
    filtered_df = DF.copy()

    # Apply filters based on drilldown state
    if selected_branch:
        filtered_df = filtered_df[filtered_df['Branch'] == selected_branch]
    if selected_brand:
        filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
    if selected_sku:
        filtered_df = filtered_df[filtered_df['SKU_Code'] == selected_sku]

    # Initialize all outputs to hidden/empty
    title = "Hierarchical Analysis"
    instruction = "Select a section from the sidebar to begin your analysis."
    fig_branch_overview = {}
    fig_branch_brands_rev = {}
    fig_branch_brands_qty = {}
    fig_brand_skus_rev = {}
    fig_brand_skus_qty = {}
    fig_customer_details = {}
    customer_purchase_details_title_style = {'display': 'none'}
    customer_purchase_details_columns = []
    customer_purchase_details_data = []

    back_to_branches_style = {'display': 'none'}
    back_to_brands_in_branch_style = {'display': 'none'}
    back_to_skus_in_brand_style = {'display': 'none'}

    if current_level == 'branch':
        title = "Overall Sales by Branch (Click to Deep Dive)"
        instruction = "Click on a branch bar to see sales details for that branch."
        
        branch_summary = filtered_df.groupby('Branch')['Redistribution Value'].sum().reset_index()
        if not branch_summary.empty: # Check if data exists before plotting
            fig_branch_overview = px.bar(branch_summary, x='Branch', y='Redistribution Value',
                                        title="Overall Sales by Branch",
                                        labels={'Redistribution Value': 'Total Revenue'})
            fig_branch_overview.update_layout(clickmode='event+select')
        else:
            fig_branch_overview = {
                'layout': {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [{
                        'text': 'No data available for Branch Overview',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 24}
                    }]
                }
            }
        
    elif current_level == 'brand':
        title = f"Sales Details for Branch: {selected_branch} (Click Brand to Deep Dive)"
        instruction = "Click on a brand bar to see SKU details for that brand within this branch."
        back_to_branches_style = {'marginBottom': '20px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '10px 15px', 'borderRadius': '5px', 'cursor': 'pointer'}
        
        brand_summary_rev = filtered_df.groupby('Brand')['Redistribution Value'].sum().nlargest(10).reset_index()
        if not brand_summary_rev.empty:
            fig_brand_rev = px.bar(brand_summary_rev, x='Brand', y='Redistribution Value',
                                title=f"Top 10 Brands in {selected_branch} by Revenue",
                                labels={'Redistribution Value': 'Total Revenue'})
            fig_brand_rev.update_layout(clickmode='event+select')
            branch_brands_chart_rev = fig_brand_rev
        else:
            branch_brands_chart_rev = {
                'layout': {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [{
                        'text': f'No revenue data for brands in {selected_branch}',
                        'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 18}
                    }]
                }
            }


        brand_qty = filtered_df.groupby('Brand')['Delivered Qty'].sum().nlargest(10).reset_index()
        if not brand_qty.empty:
            fig_brand_qty = px.bar(brand_qty, x='Brand', y='Delivered Qty',
                                title=f"Top 10 Brands in {selected_branch} by Quantity Sold",
                                labels={'Delivered Qty': 'Total Quantity'})
            branch_brands_chart_qty = fig_brand_qty
        else:
            branch_brands_chart_qty = {
                'layout': {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [{
                        'text': f'No quantity data for brands in {selected_branch}',
                        'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 18}
                    }]
                }
            }

    elif current_level == 'sku':
        title = f"SKU Details for Brand: {selected_brand} in Branch: {selected_branch} (Click SKU to Deep Dive)"
        instruction = "Click on an SKU bar to see customer details for that SKU."
        back_to_brands_in_branch_style = {'marginBottom': '20px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '10px 15px', 'borderRadius': '5px', 'cursor': 'pointer'}
        
        sku_summary_rev = filtered_df.groupby('SKU_Code')['Redistribution Value'].sum().nlargest(10).reset_index()
        if not sku_summary_rev.empty:
            fig_sku_rev = px.bar(sku_summary_rev, x='SKU_Code', y='Redistribution Value',
                                title=f"Top 10 SKUs of {selected_brand} in {selected_branch} by Revenue",
                                labels={'Redistribution Value': 'Total Revenue'})
            fig_sku_rev.update_layout(clickmode='event+select')
            brand_skus_chart_rev = fig_sku_rev
        else:
            brand_skus_chart_rev = {
                'layout': {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [{
                        'text': f'No revenue data for SKUs of {selected_brand} in {selected_branch}',
                        'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 18}
                    }]
                }
            }


        sku_qty = filtered_df.groupby('SKU_Code')['Delivered Qty'].sum().nlargest(10).reset_index()
        if not sku_qty.empty:
            fig_sku_qty = px.bar(sku_qty, x='SKU_Code', y='Delivered Qty',
                                title=f"Top 10 SKUs of {selected_brand} in {selected_branch} by Quantity Sold",
                                labels={'Delivered Qty': 'Total Quantity'})
            brand_skus_chart_qty = fig_sku_qty
        else:
            brand_skus_chart_qty = {
                'layout': {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [{
                        'text': f'No quantity data for SKUs of {selected_brand} in {selected_branch}',
                        'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 18}
                    }]
                }
            }

    elif current_level == 'customer':
        title = f"Customer Details for SKU: {selected_sku} (Brand: {selected_brand}, Branch: {selected_branch})"
        instruction = "Below are the top customers for this SKU and their recent purchase details."
        back_to_skus_in_brand_style = {'marginBottom': '20px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '10px 15px', 'borderRadius': '5px', 'cursor': 'pointer'}
        
        customer_summary = filtered_df.groupby(['Customer_Phone', 'Customer_Name'])['Total_Amount_Spent'].sum().nlargest(10).reset_index()
        if not customer_summary.empty:
            customer_summary['Customer_Display'] = customer_summary['Customer_Name'] + ' (' + customer_summary['Customer_Phone'].astype(str) + ')'
            fig_customer_spend = px.bar(customer_summary, x='Total_Amount_Spent', y='Customer_Display', orientation='h',
                                        title=f"Top 10 Customers for SKU {selected_sku}",
                                        labels={'Total_Amount_Spent': 'Total Amount Spent', 'Customer_Display': 'Customer'})
            fig_customer_spend.update_layout(yaxis={'categoryorder':'total ascending'})
            customer_details_chart = fig_customer_spend
        else:
            customer_details_chart = {
                'layout': {
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False},
                    'annotations': [{
                        'text': f'No customer data for SKU {selected_sku}',
                        'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 18}
                    }]
                }
            }

        # Detailed purchases for the selected SKU by these customers
        customer_purchase_details = filtered_df[['Customer_Phone', 'Customer_Name', 'Delivered_date', 'Delivered Qty', 'Total_Amount_Spent']].sort_values('Delivered_date', ascending=False)
        if not customer_purchase_details.empty:
            customer_purchase_details['Delivered_date'] = customer_purchase_details['Delivered_date'].dt.strftime('%Y-%m-%d')
            customer_purchase_details_title_style = {'display': 'block'}
            customer_purchase_details_columns = [{"name": i, "id": i} for i in customer_purchase_details.columns]
            customer_purchase_details_data = customer_purchase_details.to_dict('records')
        else:
            customer_purchase_details_title_style = {'display': 'none'}
            customer_purchase_details_columns = []
            customer_purchase_details_data = []


    return title, instruction, \
           fig_branch_overview, \
           branch_brands_chart_rev, branch_brands_chart_qty, \
           brand_skus_chart_rev, brand_skus_chart_qty, \
           customer_details_chart, \
           customer_purchase_details_title_style, customer_purchase_details_columns, customer_purchase_details_data, \
           back_to_branches_style, back_to_brands_in_branch_style, back_to_skus_in_brand_style


@app.callback(
    Output('drilldown-state', 'data', allow_duplicate=True), # Added allow_duplicate=True
    Input('branch-overview-chart', 'clickData'),
    Input('branch-brands-chart', 'clickData'),
    Input('brand-skus-chart', 'clickData'),
    Input('back-to-branches', 'n_clicks'),
    Input('back-to-brands-in-branch', 'n_clicks'),
    Input('back-to-skus-in-brand', 'n_clicks'),
    State('drilldown-state', 'data'),
    prevent_initial_call=True
)
def handle_hierarchical_drilldown(click_data_branch, click_data_brand, click_data_sku,
                                  n_clicks_back_branch, n_clicks_back_brand, n_clicks_back_sku,
                                  current_drilldown_state):
    """
    Manages the drill-down state for the hierarchical analysis section.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update # Use no_update when no relevant input is triggered

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    drilldown_state = current_drilldown_state.copy()

    # Handle Drill-Down
    if triggered_id == 'branch-overview-chart' and click_data_branch:
        selected_branch = click_data_branch['points'][0]['x']
        drilldown_state.update({
            'level': 'brand',
            'selected_branch': selected_branch,
            'selected_brand': None,
            'selected_sku': None,
            'selected_customer': None
        })
    elif triggered_id == 'branch-brands-chart' and click_data_brand:
        selected_brand = click_data_brand['points'][0]['x']
        drilldown_state.update({
            'level': 'sku',
            'selected_brand': selected_brand,
            'selected_sku': None,
            'selected_customer': None
        })
    elif triggered_id == 'brand-skus-chart' and click_data_sku:
        selected_sku = click_data_sku['points'][0]['x']
        drilldown_state.update({
            'level': 'customer',
            'selected_sku': selected_sku,
            'selected_customer': None # No further drill-down from customer in this flow
        })

    # Handle Drill-Up (Back Buttons)
    elif triggered_id == 'back-to-branches' and n_clicks_back_branch and n_clicks_back_branch > 0:
        drilldown_state.update({
            'level': 'branch',
            'selected_branch': None,
            'selected_brand': None,
            'selected_sku': None,
            'selected_customer': None
        })
    elif triggered_id == 'back-to-brands-in-branch' and n_clicks_back_brand and n_clicks_back_brand > 0:
        drilldown_state.update({
            'level': 'brand',
            'selected_brand': None, # Clear selected brand when going back to brand overview
            'selected_sku': None,
            'selected_customer': None
        })
    elif triggered_id == 'back-to-skus-in-brand' and n_clicks_back_sku and n_clicks_back_sku > 0:
        drilldown_state.update({
            'level': 'sku',
            'selected_sku': None, # Clear selected SKU when going back to SKU overview
            'selected_customer': None
        })
    else:
        return no_update # No relevant action, prevent update

    return drilldown_state


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
