import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import CodeAgent, LiteLLMModel, Tool
import datetime
import io
from itertools import combinations
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sales AI Agent", layout="wide")

st.title("📊 Sales Intelligence AI Agent")
st.markdown("Ask me anything about your sales data from August 2024 to January 2025!")

# --- Data Loading (with Streamlit Caching) ---
@st.cache_data
def load_sales_data_for_agent():
    try:
        df_loaded = pd.read_csv("data_sample_analysis.csv", encoding='latin1')
        # ... (rest of your df loading and preprocessing logic) ...
        df_loaded['Redistribution Value'] = df_loaded['Redistribution Value'].str.replace(',', '', regex=False).astype(float)
        df_loaded['Delivered_date'] = pd.to_datetime(df_loaded['Delivered_date'], errors='coerce', dayfirst=True)
        df_loaded['Month'] = df_loaded['Delivered_date'].dt.to_period('M')
        df_loaded['Delivered Qty'] = df_loaded['Delivered Qty'].fillna(0)
        df_loaded['Total_Amount_Spent'] = df_loaded['Redistribution Value'] * df_loaded['Delivered Qty']
        if 'Order_Id' not in df_loaded.columns:
            df_loaded['Order_Id'] = df_loaded['Customer_Phone'].astype(str) + '_' + \
                                    df_loaded['Delivered_date'].dt.strftime('%Y%m%d%H%M%S') + '_' + \
                                    df_loaded.groupby(['Customer_Phone', 'Delivered_date']).cumcount().astype(str)
        return df_loaded
    except Exception as e:
        st.error(f"Error loading sales data: {e}. Please ensure 'data_sample_analysis.csv' is available.")
        return pd.DataFrame()

@st.cache_data
def load_model_preds_for_agent():
    try:
        preds_loaded = pd.read_csv(
            "purchase_predictions_major.csv",
            parse_dates=["last_purchase_date", "pred_next_date"],
        )
        # ... (rest of your PRED_DF loading and preprocessing logic) ...
        preds_loaded = preds_loaded.rename(columns={
            "pred_next_brand":     "Next Brand Purchase",
            "pred_next_date":      "Next Purchase Date",
            "pred_spend":          "Expected Spend",
            "pred_qty":            "Expected Quantity",
            "probability":         "Probability"
        })
        preds_loaded["Next Purchase Date"] = preds_loaded["Next Purchase Date"].dt.date
        preds_loaded["Expected Spend"] = preds_loaded["Expected Spend"].round(0).astype(int)
        preds_loaded["Expected Quantity"] = preds_loaded["Expected Quantity"].round(0).astype(int)
        preds_loaded["Probability"] = (preds_loaded["Probability"] * 100).round(1)
        def suggest(p):
            if p >= 70: return "Follow-up/Alert"
            if p >= 50: return "Cross Sell"
            return "Discount"
        preds_loaded["Suggestion"] = preds_loaded["Probability"].apply(suggest)
        return preds_loaded
    except Exception as e:
        st.warning(f"Error loading prediction data: {e}. Some prediction features may be unavailable.")
        return pd.DataFrame()

# Load dataframes
df = load_sales_data_for_agent()
PRED_DF = load_model_preds_for_agent()

if df.empty:
    st.stop() # Stop execution if main data isn't loaded

# --- Helper functions (from your Colab, needed for tools) ---
def calculate_brand_sku_pairs_internal(data_frame, type_col='Brand'):
    # ... (your existing helper function code) ...
    if 'Order_Id' not in data_frame.columns:
        data_frame['Order_Id'] = data_frame['Customer_Phone'].astype(str) + "_" + \
                                 data_frame['Delivered_date'].dt.strftime('%Y%m%d%H%M%S') + '_' + \
                                 data_frame.groupby(['Customer_Phone', 'Delivered_date']).cumcount().astype(str)
    order_items = data_frame.groupby("Order_Id")[type_col].apply(set)
    pair_counts = Counter()
    for items in order_items:
        if len(items) > 1:
            for pair in combinations(items, 2):
                pair_counts[tuple(sorted(pair))] += 1
    pair_df = pd.DataFrame(pair_counts.items(), columns=[f"{type_col}_Pair_Tuple", "Count"]).sort_values(by="Count", ascending=False)
    if type_col == 'SKU_Code':
        sku_to_brand = data_frame.groupby('SKU_Code')['Brand'].first().to_dict()
        pair_df[f"{type_col}_Pair_Formatted"] = pair_df[f"{type_col}_Pair_Tuple"].apply(
            lambda x: f"{x[0]} ({sku_to_brand.get(x[0], 'Unknown')}) & {x[1]} ({sku_to_brand.get(x[1], 'Unknown')})"
        )
    else:
        pair_df[f"{type_col}_Pair_Formatted"] = pair_df[f"{type_col}_Pair_Tuple"].apply(lambda x: f"{x[0]} & {x[1]}")
    return pair_df

# --- Tool Definitions (Copy all your Tool classes here) ---
# Example:
class HeadTool(Tool):
    name = "head"
    description = "Return the first n rows of the DataFrame df."
    inputs = {"n": {"type": "integer", "description": "Number of rows to display"}}
    output_type = "object" # pandas.DataFrame
    def forward(self, n: int):
        return df.head(n)

# ... (Copy all your other Tool classes: TailTool, DescribeTool, InfoTool, HistogramTool,
#      ScatterPlotTool, CorrelationTool, PivotTableTool, FilterRowsTool, GroupByAggregateTool,
#      SortTool, TopNTool, CrosstabTool, LinRegEvalTool, PredictLinearTool, RFClassifyTool,
#      FinalAnswerTool, InsightsTool, PlotBarChartTool, PlotLineChartTool,
#      PlotDualAxisLineChartTool, BrandSKUPairAnalysisTool, CustomerProfileReportTool,
#      HeuristicNextPurchasePredictionTool, SKURecommenderTool) ...

# Ensure the SKURecommenderTool and its dependencies are correctly defined.
# The SKURecommenderTool's forward method will now use the globally loaded `df` and `PRED_DF`
# and the `calculate_brand_sku_pairs_internal` helper.
class SKURecommenderTool(Tool):
    name = "sku_recommender"
    description = (
        "Generates personalized SKU recommendations for a customer based on a hybrid model. "
        "Returns a string summary of previously purchased and recommended SKUs."
    )
    inputs = {
        "customer_phone": {"type": "string", "description": "The phone number of the customer to recommend for."},
        "top_n": {"type": "integer", "description": "Number of top recommendations to return (default 5).", "required": False, "nullable": True},
    }
    output_type = "string"

    def forward(self, customer_phone: str, top_n: int = 5):
        try:
            user_item_matrix = df.pivot_table(index='Customer_Phone', columns='SKU_Code',
                                               values='Redistribution Value', aggfunc='sum', fill_value=0)
            item_similarity = cosine_similarity(user_item_matrix.T)
            item_similarity_df = pd.DataFrame(item_similarity,
                                              index=user_item_matrix.columns,
                                              columns=user_item_matrix.columns)

            item_attributes_cols = ['SKU_Code', 'Brand']
            if 'Branch' in df.columns:
                item_attributes_cols.append('Branch')

            item_attributes = df[item_attributes_cols].drop_duplicates(subset=['SKU_Code']).set_index('SKU_Code')
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
                return "Recommender system could not be initialized: No common SKUs found between collaborative and content-based models."

            filtered_item_similarity = item_similarity_df.loc[common_skus, common_skus]
            filtered_content_similarity = content_similarity_df.loc[common_skus, common_skus]
            hybrid_similarity = (filtered_item_similarity + filtered_content_similarity) / 2

            sku_brand_map = df[['SKU_Code', 'Brand']].drop_duplicates(subset='SKU_Code').set_index('SKU_Code')

        except Exception as e:
            return f"Error preparing recommender data: {e}"

        if customer_phone not in user_item_matrix.index:
            return f"Customer {customer_phone} not found in the purchase history for recommendations."

        purchased_skus = user_item_matrix.loc[customer_phone]
        purchased_skus = purchased_skus[purchased_skus > 0].index.tolist()

        if not purchased_skus:
            return f"Customer {customer_phone} has no recorded purchases. Cannot generate recommendations."

        valid_purchased_skus = [sku for sku in purchased_skus if sku in hybrid_similarity.columns]
        if not valid_purchased_skus:
            return "No valid purchased SKUs for similarity calculation. Cannot generate recommendations."

        sku_scores = hybrid_similarity[valid_purchased_skus].mean(axis=1)
        sku_scores = sku_scores.drop(index=[s for s in purchased_skus if s in sku_scores.index], errors='ignore')

        if sku_scores.empty:
            return "No new recommendations could be generated for this customer."

        top_skus = sku_scores.sort_values(ascending=False).head(top_n)

        recommendations_df = sku_brand_map.loc[top_skus.index.intersection(sku_brand_map.index)].copy()
        recommendations_df['Similarity_Score'] = top_skus.loc[recommendations_df.index].values
        recommendations_df = recommendations_df.reset_index()

        report_parts = [f"### SKU Recommendations for Customer {customer_phone}:"]

        report_parts.append("\n**Previously Purchased SKUs:**")
        if purchased_skus:
            past_purchases_info = df[df['Customer_Phone'].astype(str) == customer_phone][['SKU_Code', 'Brand']].drop_duplicates()
            for _, row in past_purchases_info.iterrows():
                report_parts.append(f"- {row['SKU_Code']} ({row['Brand']})")
        else:
            report_parts.append("No past purchase data found for this customer.")

        report_parts.append("\n**Recommended SKUs:**")
        if not recommendations_df.empty:
            for _, row in recommendations_df.iterrows():
                report_parts.append(f"- {row['SKU_Code']} ({row['Brand']}) - Similarity: {row['Similarity_Score']:.4f}")
            report_parts.append("\n*A higher 'Similarity Score' indicates a stronger recommendation.*")
        else:
            report_parts.append("No new recommendations could be generated for this customer.")

        return "\n".join(report_parts)


# --- Instantiate all tools and create the CodeAgent ---
# (As defined in your Colab notebook, ensure all tools are instantiated)
tools = [
    HeadTool(), TailTool(), InfoTool(), DescribeTool(), HistogramTool(), ScatterPlotTool(),
    CorrelationTool(), PivotTableTool(), FilterRowsTool(), GroupByAggregateTool(), SortTool(),
    TopNTool(), CrosstabTool(), LinRegEvalTool(), PredictLinearTool(), RFClassifyTool(),
    FinalAnswerTool(), InsightsTool(),
    PlotBarChartTool(), PlotLineChartTool(), PlotDualAxisLineChartTool(),
    BrandSKUPairAnalysisTool(), CustomerProfileReportTool(),
    HeuristicNextPurchasePredictionTool(), SKURecommenderTool(),
]


# Initialize LiteLLMModel
# Use st.secrets for API key in Streamlit Cloud
# For local testing, ensure OPENROUTER_API_KEY is an environment variable
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    st.error("OpenRouter API key not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

model = LiteLLMModel(
    model_id="openrouter/meta-llama/llama-4-maverick",
    temperature=0.2,
    api_key=openrouter_api_key,
    additional_kwargs={
        "custom_llm_provider": "openrouter",
        "max_tokens": 1024,
        "max_completion_tokens": 1024,
    },
)

agent = CodeAgent(
    tools=tools,
    model=model,
    description="""
You are a Grandmaster Data Science assistant. Two pandas DataFrames are loaded:
- `df`: The main sales data, containing columns like 'Brand', 'SKU_Code', 'Customer_Name', 'Customer_Phone', 'Delivered_date', 'Redistribution Value', 'Delivered Qty', 'Order_Id', 'Month', 'Total_Amount_Spent'.
- `PRED_DF`: Contains model-based purchase predictions, with columns like 'Customer_Phone', 'Next Brand Purchase', 'Next Purchase Date', 'Expected Spend', 'Expected Quantity', 'Probability', 'Suggestion'.

You have access to these tools:
1) head(n) – Show first n rows of `df`.
2) tail(n) – Show last n rows of `df`.
3) info() – Return `df.info()` as string.
4) describe(column) – Summary stats for a column or all of `df`.
5) histogram(column, bins) – Plot histogram of a numeric column in `df`.
6) scatter_plot(column_x, column_y) – Plot scatter of two numeric columns in `df`.
7) correlation(method='pearson') – Compute correlation matrix of numeric columns in `df`.
8) pivot_table(index, columns, values, aggfunc) – Create pivot table from `df`.
9) filter_rows(column, operator, value) – Filter `df` rows. Returns the filtered DataFrame.
10) groupby_agg(group_columns, metric_column, aggfunc) – Group `df` and aggregate. Returns the aggregated DataFrame.
11) sort(column, ascending) – Sort `df` by column. Returns the sorted DataFrame.
12) top_n(metric_column, n, group_columns=None, ascending=False) – Top/Bottom n by metric (optional grouping, specify `ascending=True` for bottom N) from `df`. Returns the result DataFrame.
13) crosstab(row, column, aggfunc=None, values=None) – Crosstab between categories in `df`.
14) linreg_eval(feature_columns, target_column, test_size=0.2) – Train/test + LinearRegression on `df`, return R².
15) predict_linear(feature_columns, target_column, new_data) – Fit LinearRegression on `df`, predict new row.
16) rf_classify(feature_columns, target_column, test_size=0.2, n_estimators=100) – RF classification on `df`, return report.
17) final_answer(text) – Return a direct final answer to the user as string.
18) insights() – Compute overall sales-dataset insights and actionable recommendations. No arguments.

**New Visualization Tools:**
19) plot_bar_chart(data, x_column, y_column, title, xlabel=None, ylabel=None, horizontal=False, sort_by_x_desc=True) – Plots a bar chart from a DataFrame.
20) plot_line_chart(data, x_column, y_column, title, hue_column=None, xlabel=None, ylabel=None) – Plots a line chart for time-series data.
21) plot_dual_axis_line_chart(data, x_column, y1_column, y2_column, title, xlabel=None, y1_label=None, y2_label=None) – Plots two line charts on a dual y-axis.

**New Analysis & Reporting Tools:**
22) brand_sku_pair_analysis(type, top_n=10) – Analyzes and plots most frequently co-purchased 'Brand' or 'SKU_Code' pairs.
23) customer_profile_report(customer_phone) – Generates a comprehensive purchase report for a specific customer from `df`.
24) heuristic_next_purchase_prediction(customer_phone) – Predicts next likely purchase (SKU level) for a customer from `df` based on historical intervals.
25) sku_recommender(customer_phone, top_n=5) – Generates personalized SKU recommendations for a customer using a hybrid model (from `df` and `PRED_DF`).

**Instructions for tool usage:**
- When the user asks for “summary,” “data insights,” “actionable recommendations,” or a general overview of performance, prioritize calling `insights()`.
- For specific data queries requiring visualization, use `groupby_agg` or `top_n` first to prepare the data, then use `plot_bar_chart`, `plot_line_chart`, or `plot_dual_axis_line_chart` to visualize the result.
- For detailed customer information, use `customer_profile_report`.
- For specific next-purchase predictions, use `heuristic_next_purchase_prediction` or refer to `PRED_DF` directly if model-based predictions are requested.
- For product recommendations, use `sku_recommender`.
- Always aim to provide actionable insights where possible.
- Otherwise, pick exactly one tool that best fits and return one line of Python calling it (using named arguments). No explanations, no extra output—just the function call.
""",
    additional_authorized_imports=["pandas", "datetime", "io", "matplotlib.pyplot", "seaborn", "numpy", "itertools", "collections", "sklearn.metrics.pairwise", "sklearn.preprocessing"]
)

# --- Streamlit UI for interaction ---
user_prompt = st.text_input("➡️ Your request:", key="user_input")

if st.button("Get Response"):
    if user_prompt:
        with st.spinner("Thinking and analyzing..."):
            full_prompt = f"""
            You are a Grandmaster Data Science assistant helping a human analyze a pandas DataFrame named `df` and `PRED_DF`.
            You have access to the following tools:
            {agent.description}

            Each tool has a defined purpose and must be called using named arguments only.
            🧠 Your task:
            Based on the user request, decide which tool best fits.
            Then return ONLY one valid Python line calling that tool:
            • ✅ Example → top_n(metric_column="revenue", n=10, group_columns="region")
            • ❌ No comments, no explanations, no extra output
            The DataFrames `df` and `PRED_DF` are already loaded and ready.
            User request: {user_prompt!r}
            Tool call:
            """
            try:
                tool_call = agent.run(full_prompt).strip()
                st.info(f"Agent chose to call: `{tool_call}`")

                # Manually create tool_dispatch for direct execution in this script
                tool_dispatch = {tool.name: tool.forward for tool in tools}

                result = eval(tool_call, globals(), tool_dispatch)

                st.subheader("Agent's Response:")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, str) and "PLOTTED" in result:
                    st.success("Plot generated successfully!")
                    # The plot is automatically shown by plt.show() within the tool
                elif isinstance(result, (pd.Series, str)):
                    st.write(result)
                else:
                    st.write(f"Result: {result}")

            except Exception as e:
                st.error(f"❌ Error during tool execution: {str(e)}")
    else:
        st.warning("Please enter a request.")

st.markdown("---")
st.markdown("For best results, be specific with your requests, e.g., 'Generate SKU recommendations for customer with phone number '8060733751'.'")