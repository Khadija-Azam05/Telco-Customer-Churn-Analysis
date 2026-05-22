# Telco Customer Churn Analysis

A Machine Learning project focused on predicting customer churn in the telecommunications industry using classification models such as Logistic Regression, Decision Trees, Random Forest, and XGBoost.

## Project Overview

Customer churn prediction is a critical business problem for telecom companies. This project analyzes customer demographics, account information, and subscribed services to identify customers likely to cancel their service.

The dataset contains **7,043 customer records** with features related to:

- Customer demographics
- Account details
- Subscription services
- Billing information

The target variable is:

- **Churn** → Indicates whether a customer left the company (`Yes` or `No`).

# Objectives

- Perform data preprocessing and feature engineering
- Build multiple machine learning classification models
- Compare model performances using evaluation metrics
- Analyze factors influencing customer churn
- Identify the best model for churn prediction

# Technologies Used

## Programming Language
- Python

## Libraries
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

# Dataset Information

The dataset includes:

| Feature Type | Examples |
|---|---|
| Demographic Information | Gender, SeniorCitizen, Partner |
| Account Information | Tenure, Contract, PaymentMethod |
| Services Subscribed | InternetService, TechSupport |
| Billing Information | MonthlyCharges, TotalCharges |

Target Variable:
- `Churn`

# Project Workflow

## 1. Data Preprocessing

- Loaded dataset using Pandas
- Converted `TotalCharges` column to numeric
- Removed rows with missing values
- Encoded target variable:
  - `Yes → 1`
  - `No → 0`
- Applied One-Hot Encoding on categorical variables
- Split dataset into training and testing sets (80-20)
- Standardized numerical features using `StandardScaler`

## 2. Baseline Model — Logistic Regression

Implemented Logistic Regression to establish baseline performance.

### Analysis Performed:
- Extracted feature coefficients
- Visualized most influential positive and negative churn factors

## 3. Decision Tree Models

### Unconstrained Decision Tree
- Used entropy criterion
- Evaluated overfitting behavior

### Pruned Decision Tree
- Applied constraints:
  - `max_depth=5`
  - Controlled leaf size
- Compared training and testing accuracy
- Visualized final decision tree

## 4. Ensemble Learning

### Random Forest (Bagging)
- Implemented using multiple decision trees
- Improved generalization performance

### XGBoost (Boosting)
- Used gradient boosting framework
- Optimized using logarithmic loss

# Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

Additional visualizations:
- Confusion Matrix
- ROC Curves
- Feature Importance Charts

# Models Compared

| Model | Type |
|---|---|
| Logistic Regression | Linear Classification |
| Decision Tree | Non-linear Classification |
| Random Forest | Bagging Ensemble |
| XGBoost | Boosting Ensemble |

# Results

The project compares all models to determine:

- Best overall classification performance
- Highest Recall score for churn detection
- Strongest ROC-AUC performance

# Visualizations Included

- Churn distribution
- Feature importance plots
- Logistic Regression coefficient chart
- Decision Tree visualization
- Confusion Matrix
- ROC Curves
