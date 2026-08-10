# Sales & Profit Analytics ML Dashboard

## 📊 Project Overview

This project analyzes sales, profit, customers, products, brands, discounts, delivery performance, and returns using Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

A Streamlit dashboard is also developed to explore the dataset, visualize business performance, evaluate regression models, and predict profit for new orders.

Live : https://sales-profit-analytics-dashboard.streamlit.app/

## 🎯 Objectives

* Understand sales and profit performance
* Analyze customers and products
* Identify high-performing categories and brands
* Analyze regional and salesperson performance
* Investigate discounts and their effect on profit
* Analyze returns and delivery performance
* Build regression models for profit prediction
* Compare Linear Regression, Decision Tree, and Random Forest
* Create an interactive Streamlit dashboard

## 📁 Project Structure

```text
sales-profit-analytics-ml/
│
├── app.py
├── data.csv
├── README.md
└── requirements.txt
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

## 📌 Dataset

The dataset contains 25,000 order records with information about:

* Orders
* Customers
* Products
* Brands
* Categories
* Sales
* Discounts
* Shipping
* Delivery
* Customer ratings
* Returns
* Profit

## 🔍 Data Analysis

The project performs:

* Dataset shape and column analysis
* Data type checking
* Missing-value analysis
* Duplicate checking
* Negative-profit investigation
* Discount analysis
* Descriptive statistics
* Feature engineering
* Exploratory Data Analysis

## ⚙️ Feature Engineering

The following features are created:

* Order Year
* Order Month
* Month Name
* Quarter
* Day Name
* Age Group
* Profit Margin
* Discount Amount
* Revenue Before Discount
* Delivery Status
* Order Size Category

## 🤖 Machine Learning

Three regression models are trained:

### 1. Linear Regression

A basic regression model used as a baseline.

### 2. Decision Tree Regressor

A tree-based model that learns relationships using decision rules.

### 3. Random Forest Regressor

An ensemble model consisting of multiple decision trees.

## 🎯 Target Variable

The target variable is:

```text
Profit
```

The models predict the expected profit of an order.

## 🚫 Data Leakage Prevention

The following columns are excluded from the initial profit prediction:

```text
Order_ID
Customer_ID
Customer_Name
Customer_Rating
Returned
Total_Sale
```

Identifier and personal-name columns do not provide meaningful predictive information.

`Customer_Rating` and `Returned` may only become available after the order process.

`Total_Sale` can contain final transaction information and therefore may not be available when predicting profit before order completion.

## 📏 Model Evaluation

The models are evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

Lower MAE, MSE and RMSE indicate better performance, while a higher R² score indicates better explanatory performance.

## 📊 Visualizations

The project includes:

* Monthly Sales
* Monthly Profit
* Sales by Category
* Profit by Category
* Top Products
* Top Brands
* Sales by Region
* Profit by Region
* Salesperson Performance
* Discount vs Profit
* Delivery Days vs Customer Rating
* Return Rate by Category
* Payment Method Distribution
* Actual vs Predicted Profit
* Residual Distribution
* Feature Importance
* Model Comparison

## 🖥️ Streamlit Application

The Streamlit application contains the following sections:

### 🏠 Home

Provides:

* Project title
* Business problem
* Project objectives
* Dataset description
* Student information

### 🔎 Dataset Explorer

Provides:

* Dataset preview
* Row selector
* Region filter
* Product category filter
* Product filter
* Date filter
* Filtered data download

### 💰 Sales Dashboard

Displays:

* Total Sales
* Total Profit
* Total Orders
* Average Order Value
* Average Profit
* Return Rate
* Average Customer Rating
* Average Delivery Days

### 📦 Product & Regional Analysis

Includes:

* Sales by category
* Profit by category
* Top products
* Top brands
* Regional sales
* Regional profit
* Salesperson performance

### 👥 Customer & Delivery Analysis

Includes:

* Sales by gender
* Sales by age group
* Customer rating distribution
* Delivery analysis
* Return analysis
* Top customers

### 🤖 Model Evaluation

Includes:

* Model comparison
* Regression metrics
* Actual vs predicted profit
* Residual distribution
* Feature importance

### 🔮 Profit Prediction

Users can enter order information and receive:

* Predicted profit
* Profit category
* Loss warning

Profit categories:

```text
Loss
Low Profit
Medium Profit
High Profit
```

## 💡 Business Recommendations

The analysis can be used to:

1. Focus on high-sales product categories.
2. Protect and expand high-profit categories.
3. Investigate categories with high return rates.
4. Strengthen high-performing regions.
5. Learn successful strategies from top salespeople.
6. Monitor discounts to reduce unnecessary losses.

## 🚀 Installation

Clone the repository and install the required libraries.

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Make sure `data.csv` and `app.py` are in the same folder.

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 👨‍🎓 Student Information

**Student Name:** Muhammad Arsalan khan

**Course:** Data Science
