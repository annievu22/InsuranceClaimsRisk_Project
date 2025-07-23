# 🛡️ Insurance Claims & Policyholders Risk Analysis Project
![SQL](https://img.shields.io/badge/SQL-MySQL-blue)
![Power BI](https://img.shields.io/badge/Visualization-PowerBI-purple)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Data](https://img.shields.io/badge/Data-Insurance-informational)

> A full-scope analysis of insurance claims and policyholders using SQL, Python, and Power BI to uncover patterns in claim severity, policy types, and customer profiles. Results inform better risk assessment, customer segmentation, and pricing strategy.

---

## 1. Overview

This project investigates 10,000 synthetic insurance policyholder records to understand claim patterns, premium adjustments, and demographic factors contributing to risk. Using Python for data transformation and modeling, and Power BI for visual storytelling, the project surfaces actionable insights in claim frequency and severity across regions and profiles.

The workflow includes data cleaning, feature engineering, exploratory analysis, machine learning modeling, and the creation of an interactive dashboard to support strategic underwriting and pricing decisions.

---

## 2. Business Objectives

### 2.1. Business Problem

Insurance providers face the challenge of evaluating policyholder risk to reduce claim costs while improving customer retention. This project transforms raw policy data into insights that help insurance firms:

- Identify factors influencing claim severity
- Segment customers by region, age, and lead source
- Optimize policy and premium structures
- Predict and monitor high-risk profiles

> **Key questions** addressed in this analysis:
> - How are claims distributed by severity level?
> - Which regions and age groups show higher claim risks?
> - What’s the relationship between premium amount and claims frequency?
> - How do policy types, marital status, and lead source impact claim patterns?
> - Which features are most predictive of claim severity?

### 2.2. Goal of the Dashboard

To build a visual dashboard that:

- Summarizes total claims and average adjustment amounts  
- Explores claim severity by region, age, and policy type  
- Displays relationships between premium and claim behavior  
- Provides filters for detailed analysis by region, policy, or lead source

### 2.3. Business Impact & Insights

- Over **70% of claims** were low severity, suggesting better policy targeting for low-risk users  
- **Urban regions** reported more claims than rural or suburban, indicating regional risk concentration  
- Younger age groups (20s–40s) had higher claims volume, while older groups had lower counts  
- High claim severity correlated with **higher premium amounts and fewer discounts**  
- Referral-based leads showed **higher severity** compared to agent or online leads  
- Feature importance analysis supported enhanced risk modeling using attributes like region, credit score, and discounts

---

## 3. Data Sources & Schema

The dataset contains 10,000 anonymized insurance policyholder records with demographics, premium info, and claim-related fields.

### 🔗 Dataset Links

- **Google Drive Download**  
  [📁 View Dataset (Google Drive)](https://drive.google.com/file/d/10yhmiB2mqB6itAeOMkXORS78DeG603m2/view?usp=sharing)

### 📋 Table: `synthetic_insurance_data.csv`

| Column Name                         | Description                                            |
|-------------------------------------|--------------------------------------------------------|
| Age                                 | Age of the policyholder                                |
| Is_Senior                           | Binary flag indicating senior citizen status           |
| Marital_Status                      | Marital status of policyholder                         |
| Claims_Frequency                    | Number of claims filed                                 |
| Claims_Severity                     | Severity level: Low, Medium, High                      |
| Claims_Adjustment                   | Cost adjustment due to claims                          |
| Policy_Type                         | Coverage type (Full Coverage, Liability, etc.)         |
| Premium_Amount                      | Final premium charged                                  |
| Region                              | Region of the policyholder (Urban, Rural, Suburban)    |
| Discounts, Lead_Source, etc.        | Various discount types and customer acquisition fields |
| Credit_Score, Time_to_Conversion    | Predictive factors related to risk and engagement      |

---

## 4. Tech Stack & Methodology

This project combines statistical programming with business visualization to extract meaningful insights.

### 4.1. Tech Stack

- **SQL (MySQL):** Used for filtering, aggregation, and summary statistics  
- **Python (Pandas, Seaborn, Scikit-learn):** Used for EDA, modeling, and visualizations  
- **Power BI:** Designed the final dashboard with interactive filters and charts

### 4.2. Methodology

a. **Data Cleaning:**
- Verified null values (none found)
- Checked types and transformed columns (e.g. label encoding for categorical variables)

b. **Exploratory Data Analysis:**
- Visualized claim severity distribution
- Boxplot of premium by severity
- Age histogram by severity level
- Binned age data and segmented by region

c. **Feature Engineering & Modeling:**
- Label-encoded key categorical columns
- Split data into train-test (80/20)
- Trained a **Random Forest Classifier** to predict `Claims_Severity`
- Evaluated using classification report and confusion matrix

d. **Dashboard Design (Power BI):**
- Showcased KPIs such as total claims, avg. claim, total adjustments
- Created bar charts, pie charts, histograms, and scatter plots
- Added region, policy type, and lead source filters for interactivity

---

## 5. Python Code & Exploratory Data Analysis (EDA)

This section provides a walkthrough of the Python workflow used to clean, explore, and model the insurance claims dataset prior to Power BI visualization.


The Python script performs end-to-end analysis using `pandas`, `seaborn`, and `scikit-learn`. Below is a breakdown of the key steps and logic:

---

#### 📥 **Import Libraries and Load Data**

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('synthetic_insurance_data.csv')
print("Data shape:", df.shape)
```

The script begins by importing essential Python libraries for data manipulation (`pandas`, `numpy`), visualization (`seaborn`, `matplotlib`), and machine learning (`sklearn`). The dataset is then loaded and inspected for structure.

---

#### 🔍 **Data Exploration and Visualization**

The next steps involve exploring feature distributions, correlations, and categorical breakdowns to understand claim behavior and risk segments.

```python
df.info()
df.describe()
df.isnull().sum()
```

This provides a summary of data types, descriptive statistics, and null values.

```python
sns.countplot(data=df, x='fraud_reported')
plt.title('Distribution of Fraudulent vs Non-Fraudulent Claims')
```

A bar chart is used to check the class balance of the target variable `fraud_reported`.

```python
sns.boxplot(data=df, x='fraud_reported', y='age')
```

This visual explores whether age distribution differs between fraudulent and non-fraudulent claimants.

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```

A correlation heatmap helps detect collinearity and spot predictive variables.

---

#### 🧹 **Data Cleaning and Encoding**

```python
le = LabelEncoder()
df['fraud_reported'] = le.fit_transform(df['fraud_reported'])
df['policy_state'] = le.fit_transform(df['policy_state'])
df['incident_type'] = le.fit_transform(df['incident_type'])
df['collision_type'] = le.fit_transform(df['collision_type'].fillna('Unknown'))
df['incident_severity'] = le.fit_transform(df['incident_severity'])
```

Categorical variables are converted to numeric labels for model compatibility. Missing values in `collision_type` are filled with 'Unknown' before encoding.

---

#### 🤖 **Model Training with Random Forest**

```python
X = df.drop(['fraud_reported'], axis=1)
y = df['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Features (`X`) and target (`y`) are split, and a `RandomForestClassifier` is trained to detect fraudulent claims.

---

#### 📊 **Model Evaluation**

```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

The model's performance is evaluated using a confusion matrix and classification metrics including precision, recall, and F1-score.

---

#### 📈 **Feature Importance Visualization**

```python
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances in Fraud Detection")
plt.tight_layout()
```

This final chart visualizes which features (e.g., `incident_type`, `vehicle_claim`, `age`) are most influential in detecting fraudulent claims.

---

> This Python workflow transforms the raw insurance dataset into a model-ready format, explores key trends, and builds an interpretable machine learning model—all serving as a foundation for the Power BI dashboard.


---

## 6. Power BI Dashboard

🔗 [View Dashboard Snapshot](https://github.com/annievu22/InsuranceClaimsRisk_Project/blob/main/Insurance%20Project%20-%20PowerBI%20Snapshot.png)

### Dashboard Snapshot:

![Power BI Dashboard](https://github.com/annievu22/InsuranceClaimsRisk_Project/blob/main/Insurance%20Project%20-%20PowerBI%20Snapshot.png)

### Walkthrough of Key Visuals:

- **Top KPIs (Cards):**  
  Total Claims, Avg. Claim, Total Adjustments, and Average Claim Amount

- **Claim Severity by Region (Donut + Area Chart):**  
  Compares count and severity of claims across Urban, Suburban, and Rural regions

- **Claim Severity by Age (Stacked Column Chart):**  
  Segments claim patterns by age bins and severity level

- **Claim Frequency vs. Premium (Scatter Plot):**  
  Visualizes premium trends with frequency of claims, with a trendline

- **Policy and Lead Source Filters:**  
  Allows users to slice by Policy Type, Region, and Source of Lead

The dashboard is fully interactive, helping stakeholders understand how demographic and behavioral factors influence risk.

---

## 7. Final Conclusion

This project transforms synthetic insurance data into meaningful insights using structured analysis, modeling, and visualization. From premium behavior to claim trends, it enables insurers to design better policies, improve risk assessment, and enhance profitability.

**Key business insights:**
- Low-severity claims dominate, but high-severity claims incur large adjustments  
- Urban areas and younger policyholders tend to have higher claim frequencies  
- Lead source and marital status can signal different risk levels  
- Machine learning models can help proactively flag high-risk customers

### Future Enhancements:

- Balance class distribution using SMOTE for better model generalization  
- Integrate external data (accident history, vehicle type) for broader context  
- Deploy Power BI dashboard with real-time data using Power Query or API sources

This end-to-end project demonstrates practical experience with data science, machine learning, and dashboard storytelling in the insurance industry.

