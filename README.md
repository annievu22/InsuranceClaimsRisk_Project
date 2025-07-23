# 🛡️ Insurance Claims & Policyholders Risk Analysis Project
![SQL](https://img.shields.io/badge/SQL-MySQL-blue)
![Power BI](https://img.shields.io/badge/Visualization-PowerBI-purple)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Data](https://img.shields.io/badge/Data-Insurance-informational)

> A full-scope analysis of insurance claims and policyholders using SQL, Python, and Power BI to uncover patterns in claim severity, policy types, and customer profiles. Results inform better risk assessment, customer segmentation, and pricing strategy.

---

## 1. Overview

This project investigates 10,000 synthetic insurance policyholder records to understand claim patterns, premium adjustments, and demographic factors contributing to risk. Using SQL and Python for data transformation and modeling, and Power BI for visual storytelling, the project surfaces actionable insights in claim frequency and severity across regions and profiles.

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

- **Dashboard Snapshot**  
  ![View Snapshot](https://github.com/annievu22/InsuranceClaimsRisk_Project/blob/main/Insurance%20Project%20-%20PowerBI%20Snapshot.png)

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

## 5. SQL & Python Workflow

### 5.1. Python Data Processing

- Handled 27 features with no missing values  
- Applied label encoding for: `Marital_Status`, `Prior_Insurance`, `Policy_Type`, `Region`, `Claims_Severity`, and `Source_of_Lead`  
- Prepared final dataset for model training

### 5.2. Modeling & Evaluation

```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Confusion Matrix:**
```
[[  77  128    0]
 [   0 1346    7]
 [   2  271  169]]
```

**Classification Report:**
- Medium severity claims were predicted with the highest accuracy (F1-score ~0.87)  
- High severity claims were harder to predict due to class imbalance  
- Overall model accuracy: **80%**

### 5.3. Feature Importance

```python
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
```

Top features included:
- `Credit_Score`  
- `Region`  
- `Premium_Amount`  
- `Total_Discounts`  
- `Claims_Frequency`  

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

> The dashboard is fully interactive, helping stakeholders understand how demographic and behavioral factors influence risk.

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

> This end-to-end project demonstrates practical experience with data science, machine learning, and dashboard storytelling in the insurance industry.

