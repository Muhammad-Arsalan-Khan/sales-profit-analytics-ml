# ============================================================
# SALES & PROFIT ANALYTICS - STREAMLIT APPLICATION
# ============================================================
# Run:
#     streamlit run main.py
#
# Required file:
#     data.csv
#
# Install packages if needed:
#     pip install streamlit pandas numpy matplotlib seaborn scikit-learn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sales & Profit Analytics",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("sales_analytics.csv")

    # Convert date
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    # Feature Engineering
    df["Order_Year"] = df["Order_Date"].dt.year
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Month_Name"] = df["Order_Date"].dt.month_name()
    df["Quarter"] = df["Order_Date"].dt.quarter
    df["Day_Name"] = df["Order_Date"].dt.day_name()

    # Age Group
    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56+"
        ]
    )

    # Profit Margin
    df["Profit_Margin"] = np.where(
        df["Total_Sale"] != 0,
        (df["Profit"] / df["Total_Sale"]) * 100,
        0
    )

    # Discount Amount
    df["Discount_Amount"] = (
        df["Unit_Price"]
        * df["Quantity"]
        * df["Discount"]
    )

    # Revenue Before Discount
    df["Revenue_Before_Discount"] = (
        df["Unit_Price"] * df["Quantity"]
    )

    # Delivery Status
    df["Delivery_Status"] = np.where(
        df["Delivery_Days"] <= 3,
        "Fast",
        np.where(
            df["Delivery_Days"] <= 6,
            "Normal",
            "Slow"
        )
    )

    # Order Size Category
    df["Order_Size_Category"] = pd.cut(
        df["Quantity"],
        bins=[0, 1, 2, 4, np.inf],
        labels=[
            "Small",
            "Medium",
            "Large",
            "Very Large"
        ]
    )

    return df


df = load_data()


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_models(data):

    model_df = data.copy()

    # Columns removed because they are not appropriate
    # for initial profit prediction.
    remove_columns = [
        "Profit",
        "Order_ID",
        "Customer_ID",
        "Customer_Name",
        "Customer_Rating",
        "Returned",
        "Total_Sale"
    ]

    X = model_df.drop(
        columns=remove_columns,
        errors="ignore"
    )

    y = model_df["Profit"]

    # Remove date because raw datetime cannot directly
    # be used by the models.
    X = X.drop(
        columns=["Order_Date"],
        errors="ignore"
    )

    # Remove engineered date names that may create
    # unnecessary complexity.
    X = X.drop(
        columns=["Month_Name", "Day_Name"],
        errors="ignore"
    )

    # Convert categorical columns to string
    # so OneHotEncoder can process them.
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    X[categorical_cols] = X[categorical_cols].astype(str)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        ],
        remainder="passthrough"
    )

    # Models
    models = {

        "Linear Regression": Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]),

        "Decision Tree": Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])
    }

    metrics = []
    predictions = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(X_test)

        predictions[name] = pred

        mae = mean_absolute_error(
            y_test,
            pred
        )

        mse = mean_squared_error(
            y_test,
            pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            pred
        )

        metrics.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R² Score": r2
        })

    metrics_df = pd.DataFrame(metrics)

    # Best model based on lowest RMSE
    best_model_name = metrics_df.loc[
        metrics_df["RMSE"].idxmin(),
        "Model"
    ]

    return (
        models,
        metrics_df,
        predictions,
        X_test,
        y_test,
        best_model_name,
        categorical_cols,
        numerical_cols
    )


(
    models,
    metrics_df,
    predictions,
    X_test,
    y_test,
    best_model_name,
    categorical_cols,
    numerical_cols
) = train_models(df)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔎 Dataset Explorer",
        "💰 Sales Dashboard",
        "📦 Product & Regional Analysis",
        "👥 Customer & Delivery Analysis",
        "🤖 Model Evaluation",
        "🔮 Profit Prediction",
        "💡 Business Recommendations"
    ]
)


# ============================================================
# PAGE 1 - HOME
# ============================================================

if page == "🏠 Home":

    st.title("📊 Sales & Profit Analytics Dashboard")

    st.subheader("Business Problem")

    st.write(
        """
        The business wants to understand its sales, profit,
        customers, products, returns and delivery performance.
        The project uses data analysis and machine learning to
        identify important business patterns and predict profit.
        """
    )

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze sales and profit performance
    - Identify high-performing products and categories
    - Analyze customer behavior
    - Study returns and delivery performance
    - Understand the effect of discounts on profit
    - Compare different regression models
    - Predict profit for new orders
    """)

    st.subheader("Dataset Description")

    st.write(
        f"""
        The dataset contains {len(df):,} records and
        {df.shape[1]:,} columns. It contains information about
        orders, customers, products, brands, sales, discounts,
        shipping, delivery, returns and profit.
        """
    )

    st.subheader("Student Information")

    st.write("**Student Name:** Muhammad Arsalan Khan")
    st.write("**Course Name:** Data Science")

    st.info(
        "Please replace YOUR NAME and YOUR COURSE NAME "
        "with your actual information."
    )

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Date Range",
        f"{df['Order_Date'].min().date()} → "
        f"{df['Order_Date'].max().date()}"
    )


# ============================================================
# PAGE 2 - DATASET EXPLORER
# ============================================================

elif page == "🔎 Dataset Explorer":

    st.title("🔎 Dataset Explorer")

    st.write(
        "Use the filters below to explore the dataset."
    )

    # Region filter
    regions = sorted(
        df["Region"].dropna().unique().tolist()
    )

    selected_regions = st.multiselect(
        "Select Region",
        regions,
        default=regions
    )

    # Category filter
    categories = sorted(
        df["Product_Category"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_categories = st.multiselect(
        "Select Product Category",
        categories,
        default=categories
    )

    # Product filter
    products = sorted(
        df["Product_Name"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_products = st.multiselect(
        "Select Product",
        products,
        default=products
    )

    # Date filter
    min_date = df["Order_Date"].min().date()
    max_date = df["Order_Date"].max().date()

    selected_dates = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    filtered_df = df.copy()

    if selected_regions:
        filtered_df = filtered_df[
            filtered_df["Region"].isin(
                selected_regions
            )
        ]

    if selected_categories:
        filtered_df = filtered_df[
            filtered_df["Product_Category"].isin(
                selected_categories
            )
        ]

    if selected_products:
        filtered_df = filtered_df[
            filtered_df["Product_Name"].isin(
                selected_products
            )
        ]

    if len(selected_dates) == 2:

        start_date = pd.to_datetime(
            selected_dates[0]
        )

        end_date = pd.to_datetime(
            selected_dates[1]
        )

        filtered_df = filtered_df[
            filtered_df["Order_Date"].between(
                start_date,
                end_date
            )
        ]

    st.subheader("Filtered Dataset")

    # Number of rows selector
    max_rows = len(filtered_df)

    number_of_rows = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=max(5, max_rows),
        value=min(10, max_rows)
    )

    st.dataframe(
        filtered_df.head(number_of_rows),
        use_container_width=True
    )

    st.write(
        f"Showing {min(number_of_rows, len(filtered_df)):,} "
        f"rows out of {len(filtered_df):,} filtered rows."
    )

    # Download button
    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 3 - SALES DASHBOARD
# ============================================================

elif page == "💰 Sales Dashboard":

    st.title("💰 Sales Dashboard")

    total_sales = df["Total_Sale"].sum()

    total_profit = df["Profit"].sum()

    total_orders = df["Order_ID"].nunique()

    average_order_value = (
        total_sales / total_orders
    )

    average_profit = (
        total_profit / total_orders
    )

    return_rate = (
        (df["Returned"] == "Yes").mean() * 100
    )

    average_rating = (
        df["Customer_Rating"].mean()
    )

    average_delivery = (
        df["Delivery_Days"].mean()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Sales",
        f"{total_sales:,.2f}"
    )

    c2.metric(
        "Total Profit",
        f"{total_profit:,.2f}"
    )

    c3.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    c4.metric(
        "Average Order Value",
        f"{average_order_value:,.2f}"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Average Profit",
        f"{average_profit:,.2f}"
    )

    c6.metric(
        "Return Rate",
        f"{return_rate:.2f}%"
    )

    c7.metric(
        "Average Rating",
        f"{average_rating:.2f}"
    )

    c8.metric(
        "Average Delivery Days",
        f"{average_delivery:.2f}"
    )

    st.divider()

    # Monthly data
    monthly_data = (
        df.groupby(
            df["Order_Date"].dt.to_period("M")
        )
        .agg(
            Sales=("Total_Sale", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
    )

    monthly_data["Month"] = (
        monthly_data["Order_Date"]
        .astype(str)
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            monthly_data["Month"],
            monthly_data["Sales"],
            marker="o"
        )

        ax.set_title("Monthly Sales")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")

        ax.tick_params(
            axis="x",
            rotation=45
        )

        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            monthly_data["Month"],
            monthly_data["Profit"],
            marker="o",
            color="green"
        )

        ax.set_title("Monthly Profit")
        ax.set_xlabel("Month")
        ax.set_ylabel("Profit")

        ax.tick_params(
            axis="x",
            rotation=45
        )

        st.pyplot(fig)


# ============================================================
# PAGE 4 - PRODUCT & REGIONAL ANALYSIS
# ============================================================

elif page == "📦 Product & Regional Analysis":

    st.title("📦 Product & Regional Analysis")

    # Category Sales
    category_sales = (
        df.groupby("Product_Category")[
            "Total_Sale"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    # Category Profit
    category_profit = (
        df.groupby("Product_Category")[
            "Profit"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        category_sales.plot(
            kind="bar",
            ax=ax,
            color="skyblue"
        )

        ax.set_title(
            "Sales by Product Category"
        )

        ax.set_xlabel(
            "Product Category"
        )

        ax.set_ylabel(
            "Total Sales"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        category_profit.plot(
            kind="bar",
            ax=ax,
            color="green"
        )

        ax.set_title(
            "Profit by Product Category"
        )

        ax.set_xlabel(
            "Product Category"
        )

        ax.set_ylabel(
            "Total Profit"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        st.pyplot(fig)

    st.subheader("Top Products")

    top_products = (
        df.groupby("Product_Name")[
            "Total_Sale"
        ]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_products)

    st.subheader("Top Brands")

    top_brands = (
        df.groupby("Brand")[
            "Total_Sale"
        ]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_brands)

    st.subheader("Regional Analysis")

    region_sales = (
        df.groupby("Region")[
            "Total_Sale"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    region_profit = (
        df.groupby("Region")[
            "Profit"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Regional Sales")

        st.bar_chart(
            region_sales
        )

    with col2:

        st.write("### Regional Profit")

        st.bar_chart(
            region_profit
        )

    st.subheader(
        "Salesperson Performance"
    )

    salesperson = (
        df.groupby("Salesperson")[
            "Total_Sale"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(
        salesperson
    )


# ============================================================
# PAGE 5 - CUSTOMER & DELIVERY ANALYSIS
# ============================================================

elif page == "👥 Customer & Delivery Analysis":

    st.title(
        "👥 Customer & Delivery Analysis"
    )

    # Sales by gender
    gender_sales = (
        df.groupby("Customer_Gender")[
            "Total_Sale"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    st.subheader("Sales by Gender")

    st.bar_chart(
        gender_sales
    )

    # Sales by age group
    age_sales = (
        df.groupby(
            "Age_Group",
            observed=True
        )["Total_Sale"]
        .sum()
    )

    st.subheader(
        "Sales by Age Group"
    )

    st.bar_chart(
        age_sales
    )

    # Rating distribution
    st.subheader(
        "Customer Rating Distribution"
    )

    rating_counts = (
        df["Customer_Rating"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        rating_counts
    )

    # Delivery analysis
    st.subheader(
        "Delivery Time Analysis"
    )

    delivery_analysis = (
        df.groupby("Delivery_Days")
        .agg(
            Orders=("Order_ID", "count"),
            Average_Rating=(
                "Customer_Rating",
                "mean"
            ),
            Return_Rate=(
                "Returned",
                lambda x:
                (x == "Yes").mean() * 100
            )
        )
        .reset_index()
    )

    st.dataframe(
        delivery_analysis,
        use_container_width=True
    )

    st.line_chart(
        delivery_analysis.set_index(
            "Delivery_Days"
        )["Average_Rating"]
    )

    # Return analysis
    st.subheader(
        "Return Analysis"
    )

    return_counts = (
        df["Returned"]
        .value_counts()
    )

    st.bar_chart(
        return_counts
    )

    return_rate_category = (
        df.groupby("Product_Category")[
            "Returned"
        ]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .sort_values(ascending=False)
    )

    st.write(
        "### Return Rate by Category"
    )

    st.bar_chart(
        return_rate_category
    )

    # Top customers
    st.subheader(
        "Top Customers"
    )

    top_customers = (
        df.groupby(
            ["Customer_ID", "Customer_Name"]
        )["Total_Sale"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    st.dataframe(
        top_customers,
        use_container_width=True
    )


# ============================================================
# PAGE 6 - MODEL EVALUATION
# ============================================================

elif page == "🤖 Model Evaluation":

    st.title(
        "🤖 Regression Model Evaluation"
    )

    st.write(
        """
        The models predict Profit using information that can
        reasonably be available before the order is completed.
        """
    )

    st.subheader(
        "Columns excluded from initial prediction"
    )

    st.info(
        """
        Order_ID, Customer_ID and Customer_Name are identifiers
        or personal-name fields. Customer_Rating and Returned
        may only be known after the order process. Total_Sale
        may contain final transaction information. Including
        these variables could create data leakage or make the
        model unrealistic for pre-order prediction.
        """
    )

    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        metrics_df.style.format({
            "MAE": "{:.2f}",
            "MSE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R² Score": "{:.4f}"
        }),
        use_container_width=True
    )

    st.success(
        f"Best model based on lowest RMSE: "
        f"{best_model_name}"
    )

    # Model comparison chart
    st.subheader(
        "Model Comparison Chart"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=metrics_df,
        x="Model",
        y="RMSE",
        ax=ax
    )

    ax.set_title(
        "Model Comparison - RMSE"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")

    st.pyplot(fig)

    # Actual vs predicted
    st.subheader(
        "Actual vs Predicted Profit"
    )

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    selected_predictions = predictions[
        selected_model
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        y_test,
        selected_predictions,
        alpha=0.5
    )

    min_value = min(
        y_test.min(),
        selected_predictions.min()
    )

    max_value = max(
        y_test.max(),
        selected_predictions.max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="red",
        linestyle="--"
    )

    ax.set_title(
        f"Actual vs Predicted Profit - {selected_model}"
    )

    ax.set_xlabel(
        "Actual Profit"
    )

    ax.set_ylabel(
        "Predicted Profit"
    )

    st.pyplot(fig)

    # Residual plot
    st.subheader(
        "Residual Distribution"
    )

    residuals = (
        y_test - selected_predictions
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        residuals,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Residual Distribution - {selected_model}"
    )

    ax.set_xlabel(
        "Residual (Actual - Predicted)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    # Feature importance
    st.subheader(
        "Feature Importance - Tree Models"
    )

    importance_model_name = st.selectbox(
        "Select Tree Model",
        [
            "Decision Tree",
            "Random Forest"
        ]
    )

    tree_model = models[
        importance_model_name
    ]

    model_step = tree_model.named_steps[
        "model"
    ]

    importance = (
        model_step.feature_importances_
    )

    feature_names = (
        tree_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title(
        f"Top 15 Feature Importance - "
        f"{importance_model_name}"
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    st.pyplot(fig)


# ============================================================
# PAGE 7 - PROFIT PREDICTION
# ============================================================

elif page == "🔮 Profit Prediction":

    st.title(
        "🔮 Profit Prediction"
    )

    st.write(
        """
        Enter information about a new order. The selected
        machine learning model will estimate the expected profit.
        """
    )

    st.info(
        """
        Customer Rating, Returned and Total Sale are not used
        because they may only be known after the order is
        completed or may cause data leakage.
        """
    )

    prediction_model_name = st.selectbox(
        "Select Prediction Model",
        list(models.keys()),
        index=list(models.keys()).index(
            best_model_name
        )
    )

    model = models[
        prediction_model_name
    ]

    st.subheader(
        "Order Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        customer_gender = st.selectbox(
            "Customer Gender",
            sorted(
                df["Customer_Gender"]
                .dropna()
                .unique()
            )
        )

        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

        city = st.selectbox(
            "City",
            sorted(
                df["City"]
                .dropna()
                .unique()
            )
        )

        region = st.selectbox(
            "Region",
            sorted(
                df["Region"]
                .dropna()
                .unique()
            )
        )

        product_category = st.selectbox(
            "Product Category",
            sorted(
                df["Product_Category"]
                .dropna()
                .unique()
            )
        )

        product_name = st.selectbox(
            "Product Name",
            sorted(
                df["Product_Name"]
                .dropna()
                .unique()
            )
        )

        brand = st.selectbox(
            "Brand",
            sorted(
                df["Brand"]
                .dropna()
                .unique()
            )
        )

    with col2:

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=100,
            value=1
        )

        unit_price = st.number_input(
            "Unit Price",
            min_value=0.0,
            value=float(
                df["Unit_Price"].median()
            )
        )

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            max_value=1.0,
            value=float(
                df["Discount"].median()
            ),
            step=0.05
        )

        shipping_cost = st.number_input(
            "Shipping Cost",
            min_value=0.0,
            value=float(
                df["Shipping_Cost"].median()
            )
        )

        payment_method = st.selectbox(
            "Payment Method",
            sorted(
                df["Payment_Method"]
                .dropna()
                .unique()
            )
        )

        delivery_days = st.number_input(
            "Delivery Days",
            min_value=1,
            max_value=100,
            value=int(
                df["Delivery_Days"].median()
            )
        )

        salesperson = st.selectbox(
            "Salesperson",
            sorted(
                df["Salesperson"]
                .dropna()
                .unique()
            )
        )

    # Create prediction input
    if st.button(
        "🔮 Predict Profit",
        type="primary"
    ):

        # Calculate engineered features
        profit_margin_placeholder = 0

        discount_amount = (
            unit_price
            * quantity
            * discount
        )

        revenue_before_discount = (
            unit_price * quantity
        )

        if quantity <= 1:
            order_size = "Small"
        elif quantity <= 2:
            order_size = "Medium"
        elif quantity <= 4:
            order_size = "Large"
        else:
            order_size = "Very Large"

        if customer_age <= 25:
            age_group = "18-25"
        elif customer_age <= 35:
            age_group = "26-35"
        elif customer_age <= 45:
            age_group = "36-45"
        elif customer_age <= 55:
            age_group = "46-55"
        else:
            age_group = "56+"

        if delivery_days <= 3:
            delivery_status = "Fast"
        elif delivery_days <= 6:
            delivery_status = "Normal"
        else:
            delivery_status = "Slow"

        # Use median/representative values for
        # engineered fields that cannot be known exactly.
        prediction_input = pd.DataFrame({

            "Customer_Gender": [
                customer_gender
            ],

            "Customer_Age": [
                customer_age
            ],

            "City": [
                city
            ],

            "Region": [
                region
            ],

            "Product_Category": [
                product_category
            ],

            "Product_Name": [
                product_name
            ],

            "Brand": [
                brand
            ],

            "Quantity": [
                quantity
            ],

            "Unit_Price": [
                unit_price
            ],

            "Discount": [
                discount
            ],

            "Shipping_Cost": [
                shipping_cost
            ],

            "Payment_Method": [
                payment_method
            ],

            "Delivery_Days": [
                delivery_days
            ],

            "Salesperson": [
                salesperson
            ],

            "Order_Year": [
                int(
                    df["Order_Year"].median()
                )
            ],

            "Order_Month": [
                int(
                    df["Order_Month"].median()
                )
            ],

            "Quarter": [
                int(
                    df["Quarter"].median()
                )
            ],

            "Age_Group": [
                age_group
            ],

            "Profit_Margin": [
                profit_margin_placeholder
            ],

            "Discount_Amount": [
                discount_amount
            ],

            "Revenue_Before_Discount": [
                revenue_before_discount
            ],

            "Delivery_Status": [
                delivery_status
            ],

            "Order_Size_Category": [
                order_size
            ]
        })

        prediction_input[
            categorical_cols
        ] = prediction_input[
            categorical_cols
        ].astype(str)

        predicted_profit = model.predict(
            prediction_input
        )[0]

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        if predicted_profit < 0:

            profit_category = "Loss"

        elif predicted_profit < 25:

            profit_category = "Low Profit"

        elif predicted_profit < 75:

            profit_category = "Medium Profit"

        else:

            profit_category = "High Profit"

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Profit",
            f"{predicted_profit:,.2f}"
        )

        col2.metric(
            "Profit Category",
            profit_category
        )

        if predicted_profit < 0:

            st.error(
                "⚠️ Warning: The predicted profit is negative. "
                "This order may result in a loss."
            )

        elif profit_category == "Low Profit":

            st.warning(
                "⚠️ The predicted profit is relatively low."
            )

        elif profit_category == "Medium Profit":

            st.info(
                "The order is expected to generate medium profit."
            )

        else:

            st.success(
                "✅ The order is expected to generate high profit."
            )


# ============================================================
# PAGE 8 - BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "💡 Business Recommendations":

    st.title(
        "💡 Final Business Recommendations"
    )

    # Calculations
    best_sales_category = (
        df.groupby("Product_Category")[
            "Total_Sale"
        ]
        .sum()
        .idxmax()
    )

    best_profit_category = (
        df.groupby("Product_Category")[
            "Profit"
        ]
        .sum()
        .idxmax()
    )

    best_region = (
        df.groupby("Region")[
            "Profit"
        ]
        .sum()
        .idxmax()
    )

    best_city = (
        df.groupby("City")[
            "Total_Sale"
        ]
        .sum()
        .idxmax()
    )

    best_brand = (
        df.groupby("Brand")[
            "Total_Sale"
        ]
        .sum()
        .idxmax()
    )

    best_salesperson = (
        df.groupby("Salesperson")[
            "Total_Sale"
        ]
        .sum()
        .idxmax()
    )

    highest_return_category = (
        df.groupby("Product_Category")[
            "Returned"
        ]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .idxmax()
    )

    highest_return_rate = (
        df.groupby("Product_Category")[
            "Returned"
        ]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .max()
    )

    # Recommendation 1
    st.subheader(
        "Recommendation 1 — Focus on High-Sales Categories"
    )

    st.write(
        f"**Finding:** "
        f"{best_sales_category} generates the highest sales."
    )

    st.write(
        "**Evidence:** "
        "Category sales analysis and the Sales by Category "
        "visualization identify the strongest sales category."
    )

    st.write(
        f"**Recommendation:** "
        f"The company should maintain strong inventory and "
        f"marketing support for {best_sales_category}."
    )

    st.write(
        "**Expected Result:** "
        "Better product availability and stronger revenue "
        "performance."
    )

    # Recommendation 2
    st.subheader(
        "Recommendation 2 — Protect High-Profit Categories"
    )

    st.write(
        f"**Finding:** "
        f"{best_profit_category} generates the highest total profit."
    )

    st.write(
        "**Evidence:** "
        "Profit by Category analysis."
    )

    st.write(
        f"**Recommendation:** "
        f"Prioritize profitable products within "
        f"{best_profit_category} and avoid unnecessary discounts."
    )

    st.write(
        "**Expected Result:** "
        "Improved overall profit margins."
    )

    # Recommendation 3
    st.subheader(
        "Recommendation 3 — Improve High-Return Categories"
    )

    st.write(
        f"**Finding:** "
        f"{highest_return_category} has the highest return rate "
        f"at approximately {highest_return_rate:.2f}%."
    )

    st.write(
        "**Evidence:** "
        "Return Rate by Category analysis."
    )

    st.write(
        f"**Recommendation:** "
        f"Investigate product quality, descriptions, sizing "
        f"or customer expectations for {highest_return_category}."
    )

    st.write(
        "**Expected Result:** "
        "Lower return costs and improved customer satisfaction."
    )

    # Recommendation 4
    st.subheader(
        "Recommendation 4 — Strengthen the Best Region"
    )

    st.write(
        f"**Finding:** "
        f"{best_region} generates the highest total profit."
    )

    st.write(
        "**Evidence:** "
        "Regional profit analysis."
    )

    st.write(
        f"**Recommendation:** "
        f"Increase marketing and sales resources in {best_region}."
    )

    st.write(
        "**Expected Result:** "
        "Higher regional revenue and profit."
    )

    # Recommendation 5
    st.subheader(
        "Recommendation 5 — Learn from Top Performers"
    )

    st.write(
        f"**Finding:** "
        f"{best_salesperson} has the highest total sales."
    )

    st.write(
        "**Evidence:** "
        "Salesperson Performance analysis."
    )

    st.write(
        f"**Recommendation:** "
        f"Analyze the sales strategies used by {best_salesperson} "
        "and share successful practices with the wider sales team."
    )

    st.write(
        "**Expected Result:** "
        "Improved salesperson productivity and higher sales."
    )

    # Recommendation 6
    st.subheader(
        "Recommendation 6 — Monitor Discounting")
    st.write(
        "**Finding:** "
        "Discount levels vary across orders and some negative-profit "
        "orders also contain discounts."
    )

    st.write(
        "**Evidence:** "
        "Discount versus Profit analysis and negative-profit investigation."
    )

    st.write(
        "**Recommendation:** "
        "Avoid excessive discounts on low-margin products and "
        "evaluate discount profitability before applying promotions."
    )

    st.write(
        "**Expected Result:** "
        "Reduced unnecessary losses and improved profit margins."
    )

    # Final note
    st.success(
        "These recommendations should be reviewed together with "
        "the actual dashboard findings and model results before "
        "making major business decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Sales & Profit Analytics Project"
)

st.sidebar.caption(
    f"Dataset: {len(df):,} records"
)