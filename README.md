# 🛡️ Insurance Claims & Policyholders Risk Analysis Project
![Language](https://img.shields.io/badge/Language-Python-purple)
![Power BI](https://img.shields.io/badge/Visualization-PowerBI-yellow)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Data](https://img.shields.io/badge/Data-Insurance-red)

> A Python and Power BI analysis of insurance claims, uncovering key risk indicators, policyholder behavior, and claim severity patterns, to inform better risk assessment, customer segmentation, and pricing strategy.

---

## 1. Overview

This project explores insurance claims and policyholder data using Python and Power BI to identify risk patterns, customer segmentation opportunities, and policy characteristics driving claim severity. It supports strategic improvements in underwriting, pricing, and fraud detection.

Python was used to perform EDA, correlation analysis, and clustering based on customer profiles and claim history. The insights were presented in Power BI through interactive visuals to support actuarial and business decisions.

---

## 2. Business Objectives

### 2.1. Business Problem

As insurance firms strive to balance risk management with customer satisfaction, understanding claim behavior and policyholder profiles becomes essential. This analysis transforms policy and claims data into insights that inform pricing strategies, risk segmentation, and fraud detection.

> **Key questions** addressed in this analysis:
> - How are claims distributed by severity level?
> - Which regions and age groups show higher claim risks?
> - What’s the relationship between premium amount and claims frequency?
> - How do policy types, marital status, and lead source impact claim patterns?
> - Which features are most predictive of claim severity?

### 2.2. Business Impact & Insights

- Over 70% of claims were low severity, suggesting better policy targeting for low-risk users.  
- Urban regions reported more claims than rural or suburban, indicating regional risk concentration.  
- Younger age groups (20s–40s) had higher claims volume, while older groups had lower counts. 
- High claim severity correlated with higher premium amounts and fewer discounts. 
- Referral-based leads showed higher severity compared to agent or online leads.  
- Feature importance analysis supported enhanced risk modeling using attributes like region, credit score, and discounts.

---

## 3. Data Sources & Schema

The dataset contains 10,000 anonymized insurance policyholder records with demographic information, policy details, and claim history. It was used in Python and Power BI for risk segmentation and claims analysis.

### 🔗 Dataset Links

- **Google Drive Download:**  
  [📁 View Dataset (Google Drive)](https://drive.google.com/file/d/10yhmiB2mqB6itAeOMkXORS78DeG603m2/view?usp=sharing)

### 📝 Table: `synthetic_insurance_data.csv`

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

## 4. Methodology & Python Analysis

This section outlines the full pipeline of data cleaning, exploratory analysis, model training, and dashboard creation for a synthetic insurance claims dataset using Python and Power BI.

### 4.1. Data Cleaning

* **Verified null values and data types** → Ensured dataset integrity before modeling.

```python
df.info()
df.isnull().sum()
```

* **Label-encoded categorical features** → Converted string variables into numerical format.

```python
le = LabelEncoder()
df['fraud_reported'] = le.fit_transform(df['fraud_reported'])
df['policy_state'] = le.fit_transform(df['policy_state'])
df['incident_type'] = le.fit_transform(df['incident_type'])
df['collision_type'] = le.fit_transform(df['collision_type'].fillna('Unknown'))
df['incident_severity'] = le.fit_transform(df['incident_severity'])
```

> These steps enabled compatibility with machine learning models and ensured consistent formatting across features.

### 4.2. Exploratory Data Analysis (EDA)

* **Class distribution visualization** → Checked balance between fraudulent and non-fraudulent claims.

```python
sns.countplot(data=df, x='fraud_reported')
plt.title('Distribution of Fraudulent vs Non-Fraudulent Claims')
```

* **Age distribution by fraud status** → Investigated demographic patterns in fraud cases.

```python
sns.boxplot(data=df, x='fraud_reported', y='age')
```

* **Correlation heatmap** → Identified linear relationships between numeric variables.

```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
```

> These visuals helped uncover patterns in risk behavior and informed feature selection.

### 4.3. Fraud Detection Modeling

* **Train-test split and model fitting** → Applied Random Forest to classify fraudulent claims.

```python
X = df.drop(['fraud_reported'], axis=1)
y = df['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

* **Model evaluation** → Assessed accuracy using classification report and confusion matrix.

```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

* **Feature importance chart** → Highlighted variables most predictive of fraud.

```python
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances in Fraud Detection")
plt.tight_layout()
```
> The Random Forest model provided interpretable insights and a strong foundation for dashboard KPIs.

---

## 5. Power BI Dashboard Design

- This Power BI dashboard visualizes key policyholder segments, claim severity trends, and risk factors to support better underwriting and pricing strategies.

🔗 [View Full Dashboard on Power BI](https://project.novypro.com/HZ7pso)

### Dashboard Snapshot:

![Power BI Dashboard](https://github.com/annievu22/InsuranceClaimsRisk_Project/blob/main/Insurance%20Project%20-%20PowerBI%20Snapshot.png)

---

### Walkthrough of Key Visuals:

* **Top KPIs (Cards):**  
  Total Claims, Avg. Claim, Total Adjustments, and Average Claim Amount

* **Claim Severity by Region (Donut + Area Chart):**  
  Compares count and severity of claims across Urban, Suburban, and Rural regions

* **Claim Severity by Age (Stacked Column Chart):**  
  Segments claim patterns by age bins and severity level

* **Claim Frequency vs. Premium (Scatter Plot):**  
  Visualizes premium trends with frequency of claims, with a trendline

* **Policy and Lead Source Filters:**  
  Allows users to slice by Policy Type, Region, and Source of Lead

The dashboard is fully interactive, helping stakeholders understand how demographic and behavioral factors influence risk.

---

## 6. Final Conclusion

This project transforms synthetic insurance data into actionable business insights through structured data analysis, machine learning, and interactive dashboarding. By examining claim severity, customer demographics, and policy attributes, it supports smarter underwriting, fraud detection, and customer segmentation.

**Key business insights:**

* Most claims are low-severity, but high-severity claims drive significant adjustment costs
* Younger policyholders and urban regions exhibit higher claim frequency
* Lead source and marital status correlate with varying levels of risk exposure
* A Random Forest model helps flag high-risk customers for proactive intervention

**Future enhancement:**

* Applying SMOTE or similar techniques to balance class distribution for fraud prediction
* Enriching the dataset with external variables like driving history or vehicle details
* Automating data refresh and interactivity using Power Query and live API connections

Overall, this project showcases hands-on experience with insurance analytics, predictive modeling, and business-focused storytelling using Python and Power BI.
