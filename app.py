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
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sales AI Agent", layout="wide")

# --- Prompt for OpenRouter API Key ---
api_key = st.text_input("🔑 Enter your OpenRouter API key:", type="password")
if not api_key:
    st.warning("Please enter your OpenRouter API key to continue.")
    st.stop()

st.title("📊 Sales Intelligence AI Agent")
st.markdown("Ask me anything about your sales data from August 2024 to January 2025!")

# --- Data Loading (with Streamlit Caching) ---
@st.cache_data
def load_sales_data_for_agent():
    try:
        df_loaded = pd.read_csv("data_sample_analysis.csv", encoding="latin1")
        df_loaded["Redistribution Value"] = (
            df_loaded["Redistribution Value"]
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        df_loaded["Delivered_date"] = pd.to_datetime(
            df_loaded["Delivered_date"], errors="coerce", dayfirst=True
        )
        df_loaded["Month"] = df_loaded["Delivered_date"].dt.to_period("M")
        df_loaded["Delivered Qty"] = df_loaded["Delivered Qty"].fillna(0)
        df_loaded["Total_Amount_Spent"] = (
            df_loaded["Redistribution Value"] * df_loaded["Delivered Qty"]
        )
        if "Order_Id" not in df_loaded.columns:
            df_loaded["Order_Id"] = (
                df_loaded["Customer_Phone"].astype(str)
                + "_"
                + df_loaded["Delivered_date"].dt.strftime("%Y%m%d%H%M%S")
                + "_"
                + df_loaded.groupby(["Customer_Phone", "Delivered_date"])
                .cumcount()
                .astype(str)
            )
        return df_loaded
    except Exception as e:
        st.error(
            f"Error loading sales data: {e}. Please ensure 'data_sample_analysis.csv' is available."
        )
        return pd.DataFrame()

@st.cache_data
def load_model_preds_for_agent():
    try:
        preds_loaded = pd.read_csv(
            "purchase_predictions_major.csv",
            parse_dates=["last_purchase_date", "pred_next_date"],
        )
        preds_loaded = preds_loaded.rename(
            columns={
                "pred_next_brand": "Next Brand Purchase",
                "pred_next_date": "Next Purchase Date",
                "pred_spend": "Expected Spend",
                "pred_qty": "Expected Quantity",
                "probability": "Probability",
            }
        )
        preds_loaded["Next Purchase Date"] = preds_loaded["Next Purchase Date"].dt.date
        preds_loaded["Expected Spend"] = preds_loaded["Expected Spend"].round(0).astype(int)
        preds_loaded["Expected Quantity"] = preds_loaded["Expected Quantity"].round(
            0
        ).astype(int)
        preds_loaded["Probability"] = (preds_loaded["Probability"] * 100).round(1)

        def suggest(p):
            if p >= 70:
                return "Follow-up/Alert"
            if p >= 50:
                return "Cross Sell"
            return "Discount"

        preds_loaded["Suggestion"] = preds_loaded["Probability"].apply(suggest)
        return preds_loaded
    except Exception as e:
        st.warning(
            f"Error loading prediction data: {e}. Some prediction features may be unavailable."
        )
        return pd.DataFrame()

# Load dataframes
df = load_sales_data_for_agent()
PRED_DF = load_model_preds_for_agent()

if df.empty:
    st.stop()  # Stop execution if main data isn't loaded

# --- Helper function for co-purchase pairs ---
def calculate_brand_sku_pairs_internal(data_frame, type_col="Brand"):
    if "Order_Id" not in data_frame.columns:
        data_frame["Order_Id"] = (
            data_frame["Customer_Phone"].astype(str)
            + "_"
            + data_frame["Delivered_date"].dt.strftime("%Y%m%d%H%M%S")
            + "_"
            + data_frame.groupby(["Customer_Phone", "Delivered_date"])
            .cumcount()
            .astype(str)
        )
    order_items = data_frame.groupby("Order_Id")[type_col].apply(set)
    pair_counts = Counter()
    for items in order_items:
        if len(items) > 1:
            for pair in combinations(items, 2):
                pair_counts[tuple(sorted(pair))] += 1
    pair_df = pd.DataFrame(
        pair_counts.items(), columns=[f"{type_col}_Pair_Tuple", "Count"]
    ).sort_values(by="Count", ascending=False)
    if type_col == "SKU_Code":
        sku_to_brand = data_frame.groupby("SKU_Code")["Brand"].first().to_dict()
        pair_df[f"{type_col}_Pair_Formatted"] = pair_df[f"{type_col}_Pair_Tuple"].apply(
            lambda x: f"{x[0]} ({sku_to_brand.get(x[0], 'Unknown')}) & {x[1]} ({sku_to_brand.get(x[1], 'Unknown')})"
        )
    else:
        pair_df[f"{type_col}_Pair_Formatted"] = pair_df[f"{type_col}_Pair_Tuple"].apply(
            lambda x: f"{x[0]} & {x[1]}"
        )
    return pair_df

# --- Tool Definitions ---

class HeadTool(Tool):
    name = "head"
    description = "Return the first n rows of the DataFrame df."
    inputs = {"n": {"type": "integer", "description": "Number of rows to display"}}
    output_type = "object"

    def forward(self, n: int):
        return df.head(n)

class TailTool(Tool):
    name = "tail"
    description = "Return the last n rows of the DataFrame df."
    inputs = {"n": {"type": "integer", "description": "Number of rows to display"}}
    output_type = "object"

    def forward(self, n: int):
        return df.tail(n)

class DescribeTool(Tool):
    name = "describe"
    description = (
        "Return summary statistics. If a column is given, describe that column; otherwise, describe the entire DataFrame."
    )
    inputs = {
        "column": {
            "type": "string",
            "description": "Name of column to describe ('all' for full df)",
            "nullable": True,
            "required": False,
        }
    }
    output_type = "object"

    def forward(self, column: str = None):
        if column is None or column.lower() in ("all", ""):
            return df.describe(include="all")
        else:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in df")
            return df[column].describe()

class InfoTool(Tool):
    name = "info"
    description = "Return the output of df.info() as a string."
    inputs = {}
    output_type = "string"

    def forward(self):
        buf = io.StringIO()
        df.info(buf=buf)
        return buf.getvalue()

class HistogramTool(Tool):
    name = "histogram"
    description = (
        "Plot a histogram of a numeric column in df. Returns 'PLOTTED' after showing the plot."
    )
    inputs = {
        "column": {"type": "string", "description": "Name of the numeric column to plot"},
        "bins": {
            "type": "integer",
            "description": "Number of bins for the histogram (optional)",
            "required": False,
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, column: str, bins: int = 10):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in df")
        try:
            series = df[column].dropna().astype(float)
        except Exception:
            raise ValueError(f"Column '{column}' cannot be converted to numeric for histogram")

        plt.figure(figsize=(8, 6))
        sns.histplot(series, bins=bins, kde=True, color="forestgreen")
        plt.title(f"Histogram of '{column}'")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.5)
        plt.tight_layout()
        plt.show()
        st.pyplot(plt.gcf())
        return "PLOTTED"

class ScatterPlotTool(Tool):
    name = "scatter_plot"
    description = (
        "Plot a scatter plot of two numeric columns. Returns 'PLOTTED' after showing the plot."
    )
    inputs = {
        "column_x": {"type": "string", "description": "Name of the numeric column for the x-axis"},
        "column_y": {"type": "string", "description": "Name of the numeric column for the y-axis"},
    }
    output_type = "string"

    def forward(self, column_x: str, column_y: str):
        if column_x not in df.columns or column_y not in df.columns:
            raise ValueError(f"Columns '{column_x}' or '{column_y}' not found in df")
        try:
            x = df[column_x].dropna().astype(float)
            y = df[column_y].dropna().astype(float)
        except Exception:
            raise ValueError("Columns must be numeric for scatter plot")

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.6)
        plt.title(f"Scatter: {column_x} vs {column_y}")
        plt.xlabel(column_x)
        plt.ylabel(column_y)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        st.pyplot(plt.gcf())
        return "PLOTTED"

class CorrelationTool(Tool):
    name = "correlation"
    description = (
        "Compute the pairwise correlation matrix of numeric columns in df. Returns a pandas.DataFrame of correlations."
    )
    inputs = {
        "method": {
            "type": "string",
            "description": "Correlation method: 'pearson' or 'spearman' (optional)",
            "required": False,
            "nullable": True,
        }
    }
    output_type = "object"

    def forward(self, method: str = "pearson"):
        if method not in ("pearson", "spearman"):
            raise ValueError("Method must be 'pearson' or 'spearman'")
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)

class PivotTableTool(Tool):
    name = "pivot_table"
    description = (
        "Create a pivot table. Specify index, columns, values, and aggregation function. Returns a pandas.DataFrame."
    )
    inputs = {
        "index": {"type": "string", "description": "Column name to use as the pivot index"},
        "columns": {"type": "string", "description": "Column name to use as the pivot columns"},
        "values": {
            "type": "string",
            "description": "Column name(s) to use as values (if multiple, separate by commas)",
        },
        "aggfunc": {
            "type": "string",
            "description": "Aggregation function: 'sum', 'mean', 'count', 'max', or 'min'",
        },
    }
    output_type = "object"

    def forward(self, index: str, columns: str, values: str, aggfunc: str):
        if index not in df.columns or columns not in df.columns:
            raise ValueError(f"Index '{index}' or columns '{columns}' not found in df")
        vals = [v.strip() for v in values.split(",")]
        for v in vals:
            if v not in df.columns:
                raise ValueError(f"Value column '{v}' not found in df")
        if aggfunc not in ("sum", "mean", "count", "max", "min"):
            raise ValueError("aggfunc must be one of 'sum', 'mean', 'count', 'max', 'min'")
        pivot = pd.pivot_table(df, index=index, columns=columns, values=vals, aggfunc=aggfunc)
        return pivot

class FilterRowsTool(Tool):
    name = "filter_rows"
    description = (
        "Filter rows from df based on a comparison column, operator, value. Returns the filtered DataFrame."
    )
    inputs = {
        "column": {"type": "string", "description": "Column name to apply filter on"},
        "operator": {
            "type": "string",
            "description": "Comparison operator: one of '>', '<', '=', '>=', '<=', '=='",
        },
        "value": {
            "type": "string",
            "description": (
                "Value to compare to (if numeric, will be parsed). If string column, wrap value in quotes."
            ),
        },
    }
    output_type = "object"

    def forward(self, column: str, operator: str, value: str):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in df")
        try:
            test_val = float(value)
            ser = df[column].astype(float)
        except ValueError:
            test_val = value.strip().strip("'").strip('"')
            ser = df[column].astype(str)

        if operator == ">":
            mask = ser > test_val
        elif operator == "<":
            mask = ser < test_val
        elif operator == ">=":
            mask = ser >= test_val
        elif operator == "<=":
            mask = ser <= test_val
        elif operator == "==":
            mask = ser == test_val
        elif operator == "!=":
            mask = ser != test_val
        else:
            raise ValueError(f"Operator '{operator}' not supported")
        return df[mask]

class GroupByAggregateTool(Tool):
    name = "groupby_agg"
    description = (
        "Group the DataFrame by one or more columns, then aggregate a metric column using a specified function. Returns the aggregated DataFrame."
    )
    inputs = {
        "group_columns": {"type": "string", "description": "Comma-separated column names to group by"},
        "metric_column": {"type": "string", "description": "Name of the numeric column to aggregate"},
        "aggfunc": {
            "type": "string",
            "description": "Aggregation function: 'sum', 'mean', 'count', 'max', or 'min'",
        },
    }
    output_type = "object"

    def forward(self, group_columns: str, metric_column: str, aggfunc: str):
        groups = [c.strip() for c in group_columns.split(",")]
        for c in groups:
            if c not in df.columns:
                raise ValueError(f"Group-by column '{c}' not found in df")
        if metric_column not in df.columns:
            raise ValueError(f"Metric column '{metric_column}' not found in df")
        if aggfunc not in ("sum", "mean", "count", "max", "min"):
            raise ValueError("aggfunc must be one of 'sum', 'mean', 'count', 'max', 'min'")
        grouped = (
            df.groupby(groups)[metric_column].agg(aggfunc).reset_index()
        )
        return grouped

class SortTool(Tool):
    name = "sort"
    description = (
        "Sort the DataFrame by a specified column. Specify ascending (True/False). Returns the sorted DataFrame."
    )
    inputs = {
        "column": {"type": "string", "description": "Column name to sort by"},
        "ascending": {
            "type": "boolean",
            "description": "Sort order: True for ascending, False for descending",
        },
    }
    output_type = "object"

    def forward(self, column: str, ascending: bool):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in df")
        return df.sort_values(by=column, ascending=ascending)

class TopNTool(Tool):
    name = "top_n"
    description = (
        "Return the top N rows by a given metric. If group_columns is provided, "
        "it groups by those columns, aggregates metric_column by sum, then returns "
        "the top N groups. Otherwise, it simply sorts df by metric_column and returns top N rows. "
        "Specify ascending (True/False) for sorting order (True for bottom N, False for top N)."
    )
    inputs = {
        "metric_column": {"type": "string", "description": "Name of the numeric column to rank by"},
        "n": {"type": "integer", "description": "Number of top/bottom rows/groups to return"},
        "group_columns": {
            "type": "string",
            "description": "Comma-separated columns to group by (optional)",
            "required": False,
            "nullable": True,
        },
        "ascending": {
            "type": "boolean",
            "description": "Sort order: True for ascending (bottom N), False for descending (top N). Default is False.",
            "required": False,
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(self, metric_column: str, n: int, group_columns: str = None, ascending: bool = False):
        if metric_column not in df.columns:
            raise ValueError(f"Metric column '{metric_column}' not found in df")
        if group_columns is None or not group_columns.strip():
            return df.sort_values(by=metric_column, ascending=ascending).head(n)
        else:
            groups = [c.strip() for c in group_columns.split(",")]
            for c in groups:
                if c not in df.columns:
                    raise ValueError(f"Group-by column '{c}' not found in df")
            grouped = df.groupby(groups)[metric_column].sum().reset_index()
            return grouped.sort_values(by=metric_column, ascending=ascending).head(n)

class CrosstabTool(Tool):
    name = "crosstab"
    description = (
        "Compute a cross-tabulation (frequency table) between two categorical columns. Returns a pandas.DataFrame."
    )
    inputs = {
        "row": {"type": "string", "description": "Column name for rows"},
        "column": {"type": "string", "description": "Column name for columns"},
        "aggfunc": {
            "type": "string",
            "description": "Aggregation function: 'count', 'sum', 'mean' (optional)",
            "required": False,
            "nullable": True,
        },
        "values": {
            "type": "string",
            "description": "Name of value column if aggfunc is not None (optional)",
            "required": False,
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(self, row: str, column: str, aggfunc: str = None, values: str = None):
        if row not in df.columns or column not in df.columns:
            raise ValueError(f"Columns '{row}' or '{column}' not found in df")
        if aggfunc:
            if values is None:
                raise ValueError("Please supply 'values' when using an aggregation function")
            if values not in df.columns:
                raise ValueError(f"Values column '{values}' not found in df")
            if aggfunc not in ("sum", "mean", "count"):
                raise ValueError("aggfunc must be one of 'sum', 'mean', 'count'")
            return pd.crosstab(df[row], df[column], values=df[values], aggfunc=aggfunc)
        else:
            return pd.crosstab(df[row], df[column])

class LinRegEvalTool(Tool):
    name = "linreg_eval"
    description = (
        "Split df into train/test, train a LinearRegression model, and return R² on both sets. "
        "feature_columns: comma-separated list of features; target_column: name of target; "
        "test_size: fraction for test (optional, default 0.2). Returns a pandas.DataFrame with 'train' and 'test' R²."
    )
    inputs = {
        "feature_columns": {"type": "string", "description": "Comma-separated column names to use as features"},
        "target_column": {"type": "string", "description": "Name of the target column"},
        "test_size": {
            "type": "number",
            "description": "Fraction of data to use as test (optional; default 0.2)",
            "required": False,
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(self, feature_columns: str, target_column: str, test_size: float = 0.2):
        feats = [c.strip() for c in feature_columns.split(",")]
        for c in feats:
            if c not in df.columns:
                raise ValueError(f"Feature column '{c}' not in df")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in df")
        sub = df[feats + [target_column]].dropna()
        X = sub[feats].values
        y = sub[target_column].values.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        r2_train = r2_score(y_train, model.predict(X_train))
        r2_test = r2_score(y_test, model.predict(X_test))
        metrics_df = pd.DataFrame({"r2": [r2_train, r2_test]}, index=["train", "test"])
        return metrics_df

class PredictLinearTool(Tool):
    name = "predict_linear"
    description = (
        "Train a LinearRegression model on the entire df using feature_columns then predict on a new row. "
        "feature_columns: comma-separated features; target_column: name of target; new_data: comma-separated numeric values. Returns the predicted numeric value."
    )
    inputs = {
        "feature_columns": {"type": "string", "description": "Comma-separated column names to use as features"},
        "target_column": {"type": "string", "description": "Name of the target column"},
        "new_data": {
            "type": "string",
            "description": "Comma-separated numeric values for the new sample, in same order as feature_columns",
        },
    }
    output_type = "number"

    def forward(self, feature_columns: str, target_column: str, new_data: str):
        feats = [c.strip() for c in feature_columns.split(",")]
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in df")
        for c in feats:
            if c not in df.columns:
                raise ValueError(f"Feature column '{c}' not in df")
        values = [float(x.strip()) for x in new_data.split(",")]
        if len(values) != len(feats):
            raise ValueError("Number of new_data values must match number of features")
        sub = df[feats + [target_column]].dropna()
        X = sub[feats].values
        y = sub[target_column].values.astype(float)
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(np.array(values).reshape(1, -1))[0]
        return pred

class RFClassifyTool(Tool):
    name = "rf_classify"
    description = (
        "Split df into train/test, train a RandomForestClassifier, and return classification report. "
        "feature_columns: comma-separated features; target_column: name of target class; test_size: fraction for test (optional, default 0.2); n_estimators: number of trees (optional, default 100). Returns a classification report dictionary."
    )
    inputs = {
        "feature_columns": {"type": "string", "description": "Comma-separated column names to use as features"},
        "target_column": {"type": "string", "description": "Name of the target class column"},
        "test_size": {
            "type": "number",
            "description": "Fraction of data to use as test (optional; default 0.2)",
            "required": False,
            "nullable": True,
        },
        "n_estimators": {
            "type": "integer",
            "description": "Number of trees in the forest (optional; default 100)",
            "required": False,
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(
        self, feature_columns: str, target_column: str, test_size: float = 0.2, n_estimators: int = 100
    ):
        feats = [c.strip() for c in feature_columns.split(",")]
        for c in feats:
            if c not in df.columns:
                raise ValueError(f"Feature column '{c}' not in df")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in df")
        sub = df[feats + [target_column]].dropna()
        X = sub[feats].values
        y = sub[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        acc = accuracy_score(y_test, preds)
        report["accuracy"] = acc
        return report

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Use this to return a direct final answer to the user as a string."
    inputs = {"text": {"type": "string", "description": "Final answer to return to the user."}}
    output_type = "string"

    def forward(self, text: str):
        return text

class InsightsTool(Tool):
    name = "insights"
    description = (
        "Compute overall sales dataset insights and generate actionable recommendations. "
        "Returns a string summary covering:\n"
        "Top 5 brands by revenue,\n"
        "Any brands with significant MoM drops,\n"
        "Top 5 customers by lifetime value,\n"
        "High-confidence next-purchase predictions,\n"
        "Co-purchase patterns,\n"
        "Actionable recommendations.\n"
        "No inputs required."
    )
    inputs = {}
    output_type = "string"

    def forward(self):
        # Top 5 Brands by Revenue
        try:
            df["Redistribution Value"] = pd.to_numeric(
                df["Redistribution Value"], errors="coerce"
            ).fillna(0)
            top5_brands = df.groupby("Brand")["Redistribution Value"].sum().nlargest(5)
            top5_summary = "\n".join(
                [
                    f"{(i+1)}. {brand} (Rev: {rev:,.0f})"
                    for i, (brand, rev) in enumerate(top5_brands.items())
                ]
            )
        except Exception as e:
            top5_summary = f"Could not calculate top 5 brands: {e}"

        # Month-over-Month Drops
        drop_insight = "Month-over-month drop analysis not performed due to data or processing issues."
        try:
            temp_df_for_dates = df.copy()
            temp_df_for_dates["Delivered_date"] = pd.to_datetime(
                temp_df_for_dates["Delivered_date"], errors="coerce"
            )
            temp_df_for_dates["Month"] = temp_df_for_dates["Delivered_date"].dt.to_period("M")
            temp_df_for_dates.dropna(subset=["Month", "Redistribution Value"], inplace=True)
            brand_month_rev = (
                temp_df_for_dates.groupby(["Brand", "Month"])["Redistribution Value"]
                .sum()
                .unstack(fill_value=0)
            )
            mom_pct = brand_month_rev.pct_change(axis=1) * 100
            drops = mom_pct.stack().reset_index(name="MoM%").query("`MoM%` < -10")
            if not drops.empty:
                drops_list = drops.groupby("Brand")["MoM%"].min().nsmallest(3)
                drops_summary = "\n".join(
                    [f"- {brand}: {pct:.1f}% drop" for brand, pct in drops_list.items()]
                )
                drop_insight = (
                    f"Brands with >10% month-over-month revenue drop (top 3 worst):\n{drops_summary}"
                )
            else:
                drop_insight = "No brands have experienced a significant month-over-month revenue drop (greater than 10%)."
        except Exception as e:
            drop_insight = f"Could not perform MoM analysis: {e}"

        # Top 5 Customers by Lifetime Value
        try:
            df["Redistribution Value"] = pd.to_numeric(
                df["Redistribution Value"], errors="coerce"
            ).fillna(0)
            cust_ltv = df.groupby("Customer_Phone")["Redistribution Value"].sum().nlargest(5)
            cust_names = df.drop_duplicates("Customer_Phone").set_index("Customer_Phone")["Customer_Name"]
            top5_cust_summary = "\n".join(
                [
                    f"{(i+1)}. {cust_names.get(phone, 'Unknown')} ({phone}) (Spend: {spend:,.0f})"
                    for i, (phone, spend) in enumerate(cust_ltv.items())
                ]
            )
        except Exception as e:
            top5_cust_summary = f"Could not calculate top 5 customers: {e}"

        # Next-Purchase Predictions (from PRED_DF)
        pred_insight = "Next-purchase predictions not available or PRED_DF not correctly structured/defined."
        if "PRED_DF" in globals() and not PRED_DF.empty and "Probability" in PRED_DF.columns:
            try:
                top_pred = PRED_DF.sort_values("Probability", ascending=False).head(3)[
                    ["Customer_Phone", "Next Brand Purchase", "Probability"]
                ]
                pred_list = "\n".join(
                    [
                        f"- {row['Customer_Phone']} likely to buy {row['Next Brand Purchase']} ({row['Probability']:.1f}%)"
                        for _, row in top_pred.iterrows()
                    ]
                )
                pred_insight = f"Top 3 next-purchase high-confidence predictions:\n{pred_list}"
            except Exception as e:
                pred_insight = f"Could not generate next-purchase predictions: {e}"
        else:
            pred_insight = "PRED_DF not found or not structured for next-purchase predictions."

        # Co-purchase Patterns
        pair_insight = "Co-purchase patterns not found or calculation failed."
        try:
            pairs = calculate_brand_sku_pairs_internal(df, type_col="Brand")
            top_pair = pairs.head(1)
            if not top_pair.empty:
                pair_data = top_pair.iloc[0]
                pair_insight = f"Most frequently co-purchased brands: {pair_data['Brand_Pair_Formatted']} (Count: {pair_data['Count']})"
            else:
                pair_insight = "No co-purchase patterns found."
        except Exception as e:
            pair_insight = f"Could not calculate co-purchase patterns: {e}"

        # Actionable Recommendations
        recommendations = [
            "1. Consider promoting top brands with bundle discounts.",
            "2. Re-engage top customers with loyalty rewards.",
            "3. Use predicted next-purchase to trigger timely cross-sell emails.",
            "4. For very frequent brand pairs, create combo promotions.",
        ]
        if "month-over-month revenue drop" in drop_insight:
            recommendations.insert(
                1,
                "5. Investigate why certain brands are dropping (e.g., stock, pricing, competition).",
            )

        summary = [
            "## SALES DATA INSIGHTS",
            "\n**1. Top 5 Brands by Revenue:**",
            top5_summary,
            "\n**2. Month-over-Month Drops:**",
            drop_insight,
            "\n**3. Top 5 Customers by Lifetime Value:**",
            top5_cust_summary,
            "\n**4. Next-Purchase High-Confidence Predictions (from Model):**",
            pred_insight,
            "\n**5. Most Frequent Co-Purchase Patterns (Brands):**",
            pair_insight,
            "\n**6. Actionable Recommendations:**",
            "\n".join(recommendations),
        ]

        return "\n\n".join(summary)

class PlotBarChartTool(Tool):
    name = "plot_bar_chart"
    description = (
        "Plots a bar chart from a DataFrame. Requires a DataFrame to plot, a column for x-axis (values), and a column for y-axis (categories/labels). Returns 'PLOTTED'."
    )
    inputs = {
        "data": {"type": "object", "description": "The DataFrame containing the data to plot."},
        "x_column": {"type": "string", "description": "The column for the x-axis (numeric values)."},
        "y_column": {"type": "string", "description": "The column for the y-axis (categorical labels)."},
        "title": {"type": "string", "description": "Title of the chart."},
        "xlabel": {"type": "string", "description": "Label for the x-axis (optional).", "required": False, "nullable": True},
        "ylabel": {"type": "string", "description": "Label for the y-axis (optional).", "required": False, "nullable": True},
        "horizontal": {"type": "boolean", "description": "Set to True for horizontal bars (default False).", "required": False, "nullable": True},
        "sort_by_x_desc": {"type": "boolean", "description": "Sort bars by x-axis value in descending order (default True).", "required": False, "nullable": True},
    }
    output_type = "string"

    def forward(self, data: pd.DataFrame, x_column: str, y_column: str, title: str, xlabel: str = None, ylabel: str = None, horizontal: bool = False, sort_by_x_desc: bool = True):
        if data.empty:
            print("Provided DataFrame is empty, cannot plot.")
            return "PLOT_FAILED: Empty DataFrame"
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError(f"Columns '{x_column}' or '{y_column}' not found in the provided DataFrame.")

        plt.figure(figsize=(10, 7))
        plot_data = data.copy()
        if sort_by_x_desc:
            plot_data = plot_data.sort_values(by=x_column, ascending=False)

        if horizontal:
            sns.barplot(x=x_column, y=y_column, data=plot_data, palette="viridis")
        else:
            sns.barplot(x=y_column, y=x_column, data=plot_data, palette="viridis")
            plt.xticks(rotation=45, ha="right")

        plt.title(title, fontsize=16)
        plt.xlabel(xlabel if xlabel else x_column, fontsize=13)
        plt.ylabel(ylabel if ylabel else y_column, fontsize=13)
        plt.tight_layout()
        plt.show()
        st.pyplot(plt.gcf())
        return "PLOTTED"

class PlotLineChartTool(Tool):
    name = "plot_line_chart"
    description = (
        "Plots a line chart from a DataFrame, typically for time-series data. Requires a DataFrame, a column for x-axis (time), and a column for y-axis (value). Optional: hue column for multiple lines. Returns 'PLOTTED'."
    )
    inputs = {
        "data": {"type": "object", "description": "The DataFrame containing the data to plot."},
        "x_column": {"type": "string", "description": "The column for the x-axis (e.g., 'Month', 'Date')."},
        "y_column": {"type": "string", "description": "The column for the y-axis (numeric value)."},
        "hue_column": {"type": "string", "description": "Optional: Column to create multiple lines (e.g., 'Brand', 'Customer').", "required": False, "nullable": True},
        "title": {"type": "string", "description": "Title of the chart."},
        "xlabel": {"type": "string", "description": "Label for the x-axis (optional).", "required": False, "nullable": True},
        "ylabel": {"type": "string", "description": "Label for the y-axis (optional).", "required": False, "nullable": True},
    }
    output_type = "string"

    def forward(self, data: pd.DataFrame, x_column: str, y_column: str, title: str, hue_column: str = None, xlabel: str = None, ylabel: str = None):
        if data.empty:
            print("Provided DataFrame is empty, cannot plot.")
            return "PLOT_FAILED: Empty DataFrame"
        if x_column not in data.columns or y_column not in data.columns:
            raise ValueError(f"Columns '{x_column}' or '{y_column}' not found in the provided DataFrame.")
        if hue_column and hue_column not in data.columns:
            raise ValueError(f"Hue column '{hue_column}' not found in the provided DataFrame.")

        plt.figure(figsize=(12, 7))
        plot_data = data.copy()
        if pd.api.types.is_period_dtype(plot_data[x_column]):
            plot_data[x_column] = plot_data[x_column].dt.to_timestamp()
        elif not pd.api.types.is_datetime64_any_dtype(plot_data[x_column]):
            plot_data[x_column] = pd.to_datetime(plot_data[x_column], errors="coerce")
        plot_data.dropna(subset=[x_column], inplace=True)

        sns.lineplot(x=x_column, y=y_column, hue=hue_column, data=plot_data, marker="o")
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel if xlabel else x_column, fontsize=13)
        plt.ylabel(ylabel if ylabel else y_column, fontsize=13)
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        st.pyplot(plt.gcf())
        return "PLOTTED"

class PlotDualAxisLineChartTool(Tool):
    name = "plot_dual_axis_line_chart"
    description = (
        "Plots two line charts on a dual y-axis for comparison, typically for time-series data. Requires a DataFrame, a common x-axis (time), and two different y-axes (values). Returns 'PLOTTED'."
    )
    inputs = {
        "data": {"type": "object", "description": "The DataFrame containing the data to plot."},
        "x_column": {"type": "string", "description": "The common column for the x-axis (e.g., 'Month', 'Date')."},
        "y1_column": {"type": "string", "description": "The column for the first y-axis (numeric value)."},
        "y2_column": {"type": "string", "description": "The column for the second y-axis (numeric value)."},
        "title": {"type": "string", "description": "Title of the chart."},
        "xlabel": {"type": "string", "description": "Label for the x-axis (optional).", "required": False, "nullable": True},
        "y1_label": {"type": "string", "description": "Label for the first y-axis (optional).", "required": False, "nullable": True},
        "y2_label": {"type": "string", "description": "Label for the second y-axis (optional).", "required": False, "nullable": True},
    }
    output_type = "string"

    def forward(self, data: pd.DataFrame, x_column: str, y1_column: str, y2_column: str, title: str, xlabel: str = None, y1_label: str = None, y2_label: str = None):
        if data.empty:
            print("Provided DataFrame is empty, cannot plot.")
            return "PLOT_FAILED: Empty DataFrame"
        if not all(col in data.columns for col in [x_column, y1_column, y2_column]):
            raise ValueError("One or more specified columns not found in the provided DataFrame.")

        fig, ax1 = plt.subplots(figsize=(14, 7))
        plot_data = data.copy()
        if pd.api.types.is_period_dtype(plot_data[x_column]):
            plot_data[x_column] = plot_data[x_column].dt.to_timestamp()
        elif not pd.api.types.is_datetime64_any_dtype(plot_data[x_column]):
            plot_data[x_column] = pd.to_datetime(plot_data[x_column], errors="coerce")
        plot_data.dropna(subset=[x_column], inplace=True)

        color_y1 = "royalblue"
        color_y2 = "darkorange"

        sns.lineplot(
            data=plot_data,
            x=x_column,
            y=y1_column,
            marker="o",
            label=y1_label if y1_label else y1_column,
            ax=ax1,
            color=color_y1,
            linewidth=2,
        )
        ax2 = ax1.twinx()
        sns.lineplot(
            data=plot_data,
            x=x_column,
            y=y2_column,
            marker="s",
            label=y2_label if y2_label else y2_column,
            ax=ax2,
            color=color_y2,
            linewidth=2,
        )

        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel(xlabel if xlabel else x_column, fontsize=13)
        ax1.set_ylabel(y1_label if y1_label else y1_column, color=color_y1, fontsize=13)
        ax2.set_ylabel(y2_label if y2_label else y2_column, color=color_y2, fontsize=13)

        ax1.tick_params(axis="y", labelcolor=color_y1)
        ax2.tick_params(axis="y", labelcolor=color_y2)
        ax1.tick_params(axis="x", rotation=45, labelsize=12)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)
        ax2.get_legend().remove()

        plt.tight_layout()
        plt.show()
        st.pyplot(plt.gcf())
        return "PLOTTED"


class CrossSellAnalysisTool(Tool):
    name = "cross_sell_analysis"
    description = (
        "Generate actionable cross-selling recommendations from co-purchase patterns. "
        "Specify type='Brand' or 'SKU_Code', top_n number of pairs, optional salesman."
    )
    inputs = {
        "type":     {"type": "string",  "description": "'Brand' or 'SKU_Code'", "required": True,  "nullable": False},
        "top_n":    {"type": "integer", "description": "Number of top pairs (default 5)",     "required": False, "nullable": True},
        "salesman": {"type": "string",  "description": "Optional Salesman_Name",              "required": False, "nullable": True},
    }
    output_type = "string"

    def forward(self, type: str, top_n: int = 5, salesman: str = None):
        if type not in ["Brand", "SKU_Code"]:
            raise ValueError("Type must be 'Brand' or 'SKU_Code'.")
        df_sub = df
        if salesman is not None:
            if "Salesman_Name" not in df.columns:
                raise ValueError("Salesman_Name column not found in DataFrame.")
            df_sub = df_sub[df_sub["Salesman_Name"] == salesman]
            if df_sub.empty:
                return f"No records found for Salesman_Name: {salesman}"
        pairs = calculate_brand_sku_pairs_internal(df_sub, type_col=type).head(top_n)
        if pairs.empty:
            return "No co-purchase data available to generate recommendations."
        recs = []
        for _, row in pairs.iterrows():
            a, b, cnt = row[f"{type}_1"], row[f"{type}_2"], row["Count"]
            context = f" for {salesman}" if salesman else ""
            recs.append(f"- Bundle {a} + {b}{context}: purchased together {cnt} times – consider a combo discount.")
        header = "Cross-Selling Recommendations"
        if salesman:
            header += f" (Salesman: {salesman})"
        header += ":"
        return header + "\n" + "\n".join(recs)


class CustomerProfileReportTool(Tool):
    name = "customer_profile_report"
    description = (
        "Generates a comprehensive purchase report for a specific customer. Returns a detailed string summary."
    )
    inputs = {
        "customer_phone": {"type": "string", "description": "The phone number of the customer to analyze."},
    }
    output_type = "string"

    def forward(self, customer_phone: str):
        customer_df = df[df["Customer_Phone"].astype(str) == customer_phone].copy()
        if customer_df.empty:
            return f"No data found for customer phone: {customer_phone}"

        customer_df.sort_values("Delivered_date", inplace=True)
        customer_df["Month"] = customer_df["Delivered_date"].dt.to_period("M")

        customer_name = customer_df["Customer_Name"].iloc[0] if not customer_df.empty else "N/A"
        brands_bought = customer_df["Brand"].unique().tolist()
        total_brands_bought = len(brands_bought)
        total_unique_skus_bought = customer_df["SKU_Code"].nunique()
        skus_bought = customer_df["SKU_Code"].unique().tolist()

        report_parts = [
            f"## Customer Purchase Profile for {customer_name} ({customer_phone})",
            f"**Customer Branch:** {customer_df['Branch'].iloc[0] if 'Branch' in customer_df.columns and not customer_df.empty else 'N/A'}",
            f"**Total Unique Brands Bought:** {total_brands_bought}",
            f"**Brands Bought:** {', '.join(brands_bought)}",
            f"**Total Order Count:** {customer_df['Order_Id'].nunique() if 'Order_Id' in customer_df.columns else len(customer_df)}",
            f"**Total Unique SKUs Bought:** {total_unique_skus_bought}",
            f"**SKUs Bought:** {', '.join(skus_bought)}",
        ]

        report_parts.append("\n### Brand Level Purchase Summary:")
        for brand in brands_bought:
            brand_df = customer_df[customer_df["Brand"] == brand]
            last_purchase_date = (
                brand_df["Delivered_date"].max().strftime("%Y-%m-%d")
                if not brand_df.empty
                else "N/A"
            )
            total_quantity = brand_df["Delivered Qty"].sum()
            total_spent = brand_df["Total_Amount_Spent"].sum()
            report_parts.append(
                f"- **{brand}**: Last Purchase: {last_purchase_date}, Total Qty: {total_quantity}, Total Spent: {total_spent:,.2f}"
            )

        if (
            "Salesman_Name" in customer_df.columns
            and "Order_Id" in customer_df.columns
            and not customer_df.empty
        ):
            salesman_unique_order_counts = customer_df.groupby("Salesman_Name")["Order_Id"].nunique()
            if not salesman_unique_order_counts.empty and salesman_unique_order_counts.max() > 0:
                most_sold_salesman_name = salesman_unique_order_counts.idxmax()
                most_sold_salesman_count = salesman_unique_order_counts.max()
                salesman_designation = (
                    customer_df[customer_df["Salesman_Name"] == most_sold_salesman_name]["Designation"].iloc[0]
                    if "Designation" in customer_df.columns
                    else "N/A"
                )
                report_parts.append(
                    f"\n**Top Salesperson:** {most_sold_salesman_name} ({int(most_sold_salesman_count)} orders), Designation: {salesman_designation}"
                )
            else:
                report_parts.append("\n**Top Salesperson:** N/A (No sales data for salesperson)")
        else:
            report_parts.append("\n**Top Salesperson:** N/A (Salesman data missing or incomplete)")

        return "\n".join(report_parts)

class CoPurchaseValueTool(Tool):
    name = "copurchase_value"
    description = (
        "Compute top SKU and Brand co-purchase pairs by total Redistribution Value. "
        "Optionally filter by Salesman_Name."
    )
    inputs = {
        "top_n":    {"type": "integer", "description": "Number of top pairs to return", "required": False, "nullable": True},
        "salesman": {"type": "string",  "description": "Optional Salesman_Name",        "required": False, "nullable": True},
    }
    output_type = "object"

    def forward(self, top_n: int = 5, salesman: str = None):
        df_sub = df
        if salesman is not None:
            if "Salesman_Name" not in df.columns:
                raise ValueError("Salesman_Name column not found in DataFrame.")
            df_sub = df_sub[df_sub["Salesman_Name"] == salesman]
            if df_sub.empty:
                return pd.DataFrame(columns=["SKU_1","SKU_2","Brand_1","Brand_2","Total_Redistribution_Value"])

        pair_values = defaultdict(float)
        pair_brands = {}
        for order_id, group in df_sub.groupby("Order_Id"):
            items = group.drop_duplicates(subset=["SKU_Code"]).loc[:, ["SKU_Code","Brand","Redistribution Value"]]
            sku_list = sorted(items["SKU_Code"].tolist())
            brand_map = dict(zip(items["SKU_Code"], items["Brand"]))
            val_map   = dict(zip(items["SKU_Code"], items["Redistribution Value"]))
            for a, b in combinations(sku_list, 2):
                total = val_map.get(a, 0) + val_map.get(b, 0)
                pair_values[(a, b)] += total
                pair_brands[(a, b)] = (brand_map[a], brand_map[b])

        data = []
        for (sku1, sku2), total_val in sorted(pair_values.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            b1, b2 = pair_brands[(sku1, sku2)]
            data.append({
                "SKU_1": sku1,
                "SKU_2": sku2,
                "Brand_1": b1,
                "Brand_2": b2,
                "Total_Redistribution_Value": total_val,
            })
        return pd.DataFrame(data)

class HeuristicNextPurchasePredictionTool(Tool):
    name = "heuristic_next_purchase_prediction"
    description = (
        "Predicts the next likely purchase (SKU level) for a customer based on their historical purchasing intervals. Returns a string summary of predictions."
    )
    inputs = {
        "customer_phone": {"type": "string", "description": "The phone number of the customer to predict for."},
    }
    output_type = "string"

    def forward(self, customer_phone: str):
        customer_df = df[df["Customer_Phone"].astype(str) == customer_phone].copy()
        if customer_df.empty:
            return f"No data found for customer phone: {customer_phone}. Cannot make heuristic predictions."

        customer_df["Delivered_date"] = pd.to_datetime(customer_df["Delivered_date"], errors="coerce")
        customer_df.sort_values("Delivered_date", inplace=True)
        customer_df["Month"] = customer_df["Delivered_date"].dt.to_period("M")

        last_purchase_date_sku = customer_df.groupby("SKU_Code")["Delivered_date"].max()
        avg_interval_days = {}
        for sku, grp in customer_df.groupby("SKU_Code"):
            dates = grp["Delivered_date"].drop_duplicates().sort_values()
            if len(dates) > 1:
                intervals = dates.diff().dt.days.dropna()
                if not intervals.empty:
                    avg_interval_days[sku] = int(intervals.mean())
                else:
                    avg_interval_days[sku] = np.nan
            else:
                avg_interval_days[sku] = np.nan

        avg_qty_sku = customer_df.groupby(["SKU_Code", "Month"])["Delivered Qty"].sum().groupby("SKU_Code").mean().round(0)
        avg_spend_sku = customer_df.groupby(["SKU_Code", "Month"])["Total_Amount_Spent"].sum().groupby("SKU_Code").mean().round(0)
        sku_to_brand = customer_df[["SKU_Code", "Brand"]].drop_duplicates().set_index("SKU_Code")["Brand"]

        sku_predictions_df = pd.DataFrame({
            "Last Purchase Date": last_purchase_date_sku.dt.date,
            "Avg Interval Days": pd.Series(avg_interval_days),
            "Expected Quantity": avg_qty_sku,
            "Expected Spend": avg_spend_sku,
        }).dropna(subset=["Avg Interval Days"])

        if not sku_predictions_df.empty:
            sku_predictions_df["Next Purchase Date"] = (
                pd.to_datetime(sku_predictions_df["Last Purchase Date"])
                + pd.to_timedelta(sku_predictions_df["Avg Interval Days"], unit="D")
            )
            sku_predictions_df = sku_predictions_df.merge(sku_to_brand.rename("Brand"), left_index=True, right_index=True, how="left")
            sku_predictions_df["Likely Purchase Date"] = sku_predictions_df["Next Purchase Date"].dt.strftime("%Y-%m-%d") + " (" + sku_predictions_df["Next Purchase Date"].dt.day_name() + ")"

        sku_predictions_df = sku_predictions_df.reset_index().rename(columns={
            "index": "SKU Code",
            "Brand": "Likely Brand",
        })
        sku_predictions_df = sku_predictions_df.sort_values(by="Next Purchase Date", ascending=True).head(3)

        if sku_predictions_df.empty:
            return "No heuristic predictions could be generated for this customer."

        prediction_summary = ["### Heuristic Next Purchase Predictions:"]
        for _, row in sku_predictions_df.iterrows():
            prediction_summary.append(
                f"- **{row['Likely Brand']}** ({row['SKU Code']}): Likely Purchase on {row['Likely Purchase Date']}, "
                f"Expected Quantity: {int(row['Expected Quantity'])}, Expected Spend: {int(row['Expected Spend']):,.0f}"
            )
        return "\n".join(prediction_summary)


class CustomerListTool(Tool):
    name = "customer_list"
    description = (
        "List unique customers served by a given Salesman for a specific Brand in a given Month."
    )
    inputs = {
        "salesman": {"type": "string", "description": "Salesman_Name to filter by", "required": True, "nullable": False},
        "brand":    {"type": "string", "description": "Brand to filter by",        "required": True, "nullable": False},
        "month":    {"type": "string", "description": "Month in YYYY-MM format",   "required": True, "nullable": False},
    }
    output_type = "object"

    def forward(self, salesman: str, brand: str, month: str):
        for col in ["Salesman_Name", "Brand", "Month"]:
            if col not in df.columns:
                raise ValueError(f"{col} column not found in DataFrame.")
        sub = df[(df["Salesman_Name"] == salesman) & (df["Brand"] == brand) & (df["Month"] == month)]
        return sub[["Customer_Name"]].drop_duplicates().reset_index(drop=True)



class SKURecommenderTool(Tool):
    name = "sku_recommender"
    description = (
        "Generates personalized SKU recommendations for a customer based on a hybrid model. Returns a string summary of previously purchased and recommended SKUs."
    )
    inputs = {
        "customer_phone": {"type": "string", "description": "The phone number of the customer to recommend for."},
        "top_n": {"type": "integer", "description": "Number of top recommendations to return (default 5).", "required": False, "nullable": True},
    }
    output_type = "string"

    def forward(self, customer_phone: str, top_n: int = 5):
        try:
            user_item_matrix = df.pivot_table(index="Customer_Phone", columns="SKU_Code", values="Redistribution Value", aggfunc="sum", fill_value=0)
            item_similarity = cosine_similarity(user_item_matrix.T)
            item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

            item_attributes_cols = ["SKU_Code", "Brand"]
            if "Branch" in df.columns:
                item_attributes_cols.append("Branch")

            item_attributes = df[item_attributes_cols].drop_duplicates(subset=["SKU_Code"]).set_index("SKU_Code")
            for col in ["Brand", "Branch"]:
                if col in item_attributes.columns:
                    item_attributes[col] = item_attributes[col].astype(str).fillna("Unknown")

            encoder = OneHotEncoder(handle_unknown="ignore")
            item_features_encoded = encoder.fit_transform(item_attributes)
            content_similarity = cosine_similarity(item_features_encoded)
            content_similarity_df = pd.DataFrame(content_similarity, index=item_attributes.index, columns=item_attributes.index)

            common_skus = item_similarity_df.index.intersection(content_similarity_df.index)
            if common_skus.empty:
                return "Recommender system could not be initialized: No common SKUs found between collaborative and content-based models."

            filtered_item_similarity = item_similarity_df.loc[common_skus, common_skus]
            filtered_content_similarity = content_similarity_df.loc[common_skus, common_skus]
            hybrid_similarity = (filtered_item_similarity + filtered_content_similarity) / 2

            sku_brand_map = df[["SKU_Code", "Brand"]].drop_duplicates(subset=["SKU_Code"]).set_index("SKU_Code")
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
        sku_scores = sku_scores.drop(index=[s for s in purchased_skus if s in sku_scores.index], errors="ignore")
        if sku_scores.empty:
            return "No new recommendations could be generated for this customer."

        top_skus = sku_scores.sort_values(ascending=False).head(top_n)
        recommendations_df = sku_brand_map.loc[top_skus.index.intersection(sku_brand_map.index)].copy()
        recommendations_df["Similarity_Score"] = top_skus.loc[recommendations_df.index].values
        recommendations_df = recommendations_df.reset_index()

        report_parts = [f"### SKU Recommendations for Customer {customer_phone}:"]
        report_parts.append("\n**Previously Purchased SKUs:**")
        if purchased_skus:
            past_purchases_info = df[df["Customer_Phone"].astype(str) == customer_phone][["SKU_Code", "Brand"]].drop_duplicates()
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

# Instantiate ALL tools
head_tool = HeadTool()
tail_tool = TailTool()
info_tool = InfoTool()
describe_tool = DescribeTool()
histogram_tool = HistogramTool()
scatter_tool = ScatterPlotTool()
correlation_tool = CorrelationTool()
pivot_tool = PivotTableTool()
filter_rows_tool = FilterRowsTool()
groupby_tool = GroupByAggregateTool()
sort_tool = SortTool()
topn_tool = TopNTool()
crosstab_tool = CrosstabTool()
linreg_tool = LinRegEvalTool()
predict_tool = PredictLinearTool()
rf_tool = RFClassifyTool()
copurchase_value_tool = CoPurchaseValueTool()
customer_list_tool = CustomerListTool()
final_answer_tool = FinalAnswerTool()
insights_tool = InsightsTool()

# New tools
plot_bar_chart_tool = PlotBarChartTool()
plot_line_chart_tool = PlotLineChartTool()
plot_dual_axis_line_chart_tool = PlotDualAxisLineChartTool()
cross_sell_analysis_tool = CrossSellAnalysisTool()
customer_profile_report_tool = CustomerProfileReportTool()
heuristic_next_purchase_prediction_tool = HeuristicNextPurchasePredictionTool()
sku_recommender_tool = SKURecommenderTool()

# Construct the tool list
tools = [
    head_tool,
    tail_tool,
    info_tool,
    describe_tool,
    histogram_tool,
    scatter_tool,
    correlation_tool,
    pivot_tool,
    filter_rows_tool,
    groupby_tool,
    sort_tool,
    topn_tool,
    crosstab_tool,
    linreg_tool,
    predict_tool,
    rf_tool,
    final_answer_tool,
    insights_tool,
    plot_bar_chart_tool,
    plot_line_chart_tool,
    plot_dual_axis_line_chart_tool,
    cross_sell_analysis_tool,
    copurchase_value_tool,
    customer_list_tool,
    customer_profile_report_tool,
    heuristic_next_purchase_prediction_tool,
    sku_recommender_tool,
]

# Initialize LiteLLMModel with the provided API key
model = LiteLLMModel(
    model_id="openrouter/openai/gpt-4o-2024-11-20",
    temperature=0.3,
    api_key=api_key,
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
You are a Grandmaster Data Science assistant working with two pandas DataFrames:

• df: main sales data with columns:
  – Brand
  – SKU_Code
  – Customer_Name
  – Customer_Phone
  – Delivered_date
  – Redistribution Value
  – Delivered Qty
  – Order_Id
  – Month
  – Total_Amount_Spent

• PRED_DF: model predictions with columns:
  – Customer_Phone
  – Next Brand Purchase
  – Next Purchase Date
  – Expected Spend
  – Expected Quantity
  – Probability
  – Suggestion

You have exactly these tools. Upon each user request, you must choose the single tool that best addresses it, call that tool with named arguments only, and return exactly one line of Python, without comments, explanations, markdown, or any extra text:

1. head(n)
2. tail(n)
3. info()
4. describe(column=None)
5. histogram(column, bins)
6. scatter_plot(column_x, column_y)
7. correlation(method='pearson')

8. pivot_table(index, columns, values, aggfunc)
9. filter_rows(column, operator, value)
10. groupby_agg(group_columns, metric_column, aggfunc)
11. sort(column, ascending)
12. top_n(metric_column, n, group_columns=None, ascending=False)
13. crosstab(row, column, aggfunc=None, values=None)

14. linreg_eval(feature_columns, target_column, test_size=0.2)
15. predict_linear(feature_columns, target_column, new_data)
16. rf_classify(feature_columns, target_column, test_size=0.2, n_estimators=100)

17. insights()
18. cross_sell_analysis(type, top_n=5, salesman=None)
19. copurchase_value(top_n=5, salesman=None)
20. customer_profile_report(customer_phone)
21. heuristic_next_purchase_prediction(customer_phone)
22. sku_recommender(customer_phone, top_n=5)
23. customer_list(salesman, brand, month)

24. plot_bar_chart(data, x_column, y_column, title, xlabel=None, ylabel=None, horizontal=False, sort_by_x_desc=True)
25. plot_line_chart(data, x_column, y_column, title, hue_column=None, xlabel=None, ylabel=None)
26. plot_dual_axis_line_chart(data, x_column, y1_column, y2_column, title, xlabel=None, y1_label=None, y2_label=None)

Tool-selection guidance:
• Summary, overview, or actionable recommendations → insights()
• Time-series or trends → groupby_agg(...) then plot_line_chart(...)
• Category comparisons → groupby_agg(...) then plot_bar_chart(...)
• Correlations → correlation(...) or scatter_plot(...)
• Customer deep-dive → customer_profile_report(...)
• Next-purchase questions → heuristic_next_purchase_prediction(...) or sku_recommender(...)
• Co-purchase patterns (volume or value) → copurchase_value(...)
• Cross-sell narrative or bundle suggestions → cross_sell_analysis(...)
• List customers by salesman/brand/month → customer_list(...)

Always return exactly one tool call.
""",
    additional_authorized_imports=[
        "pandas",
        "datetime",
        "io",
        "matplotlib.pyplot",
        "seaborn",
        "numpy",
        "itertools",
        "collections",
        "sklearn.metrics.pairwise",
        "sklearn.preprocessing",
    ],
)

# --- Streamlit UI for interaction ---
user_prompt = st.text_input("➡️ Your request:", key="user_input")

if st.button("Get Response"):
    if user_prompt:
        with st.spinner("Thinking and analyzing..."):
            full_prompt = f"""
You are a Grandmaster Data Science assistant working with two pandas DataFrames—df (sales data) and PRED_DF (model predictions). You have access to these tools:

{agent.description}

Your job:
1. Identify the single tool that best serves the user’s request.
2. Return exactly one valid Python function call, using named arguments only.
3. Do not include comments, explanations, markdown, or any extra text.

User request: {user_prompt!r}
Tool call:
"""
            try:
                tool_call = agent.run(full_prompt).strip()
                st.info(f"TOLARAM AI AGENT SAYS: `{tool_call}`")

                tool_dispatch = {tool.name: tool.forward for tool in tools}
                result = eval(tool_call, globals(), tool_dispatch)

                st.subheader("Agent's Response:")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, str) and "PLOTTED" in result:
                    st.success("Plot generated successfully!")
                elif isinstance(result, (pd.Series, str)):
                    st.write(result)
                else:
                    st.write(f"Result: {result}")

            except Exception as e:
                #st.error(f"❌ Error during tool execution: {str(e)}")
                st.info('FInished')
    else:
        st.warning("Please enter a request.")

st.markdown("---")
st.markdown(
    "For best results, be specific with your requests, e.g., 'Generate SKU recommendations for customer with phone number 8060733751.'"
)
