# Employee Attrition Prediction System

An AI-powered HR analytics dashboard for employee attrition risk assessment.

## Project Overview

The **Employee Attrition Prediction System** is a unified machine learning and business intelligence platform designed to help Human Resources leadership proactively identify, understand, and mitigate employee turnover. Built using Streamlit, scikit-learn, and the IBM HR Analytics benchmark dataset, the system combines exploratory workforce analytics with a calibrated machine learning prediction pipeline.

---

## Key Features

1. **Workforce Overview**: Executive KPI summary cards tracking headcounts, attrition rates, distribution breakdowns, and automated data quality audits (missing values, duplicates).
2. **Attrition Analysis**: Interactive multi-dimensional visualizations examining turnover drivers across departments, overtime status, job satisfaction levels, and work-life balance ratings.
3. **Predict Employee Attrition**: Dynamic decision-support form capturing 30 employee attributes to evaluate turnover risk probability using an end-to-end trained ML pipeline.
4. **Model Performance**: Transparent audit of model architecture, preprocessing pipelines, calibrated decision thresholds, and held-out test-set evaluation metrics.

---

## Machine Learning Pipeline & Architecture

* **Algorithm**: Recall-Focused Logistic Regression with balanced class weighting (`class_weight='balanced'`) to handle class imbalance effectively.
* **Feature Processing**:
  * **23 Numerical Features**: Imputed using median statistics and scaled via standard score normalization (`StandardScaler`).
  * **7 Categorical Features**: Imputed using most frequent values and transformed via one-hot encoding (`OneHotEncoder(handle_unknown='ignore')`).
  * Integrated within a unified `ColumnTransformer` to prevent data leakage.
* **Calibrated Decision Threshold**: Set at **0.39** based on cross-validation optimization. Because the operational cost of missing an employee who departs (false negative) outweighs the cost of proactive engagement (false positive), this threshold guarantees **>80% recall** on the positive attrition class.

### Benchmark Evaluation Results (Held-Out Test Set)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 73.13% |
| **Recall (Attrition = Yes)** | 80.85% |
| **Precision (Attrition = Yes)** | 35.19% |
| **F1-Score (Attrition = Yes)** | 49.03% |
| **ROC-AUC** | 80.32% |
| **Average Precision** | 56.12% |

---

## Local Installation & Quickstart

### Prerequisites
* Python 3.10 to 3.13
* Virtual environment recommended (`venv` or `conda`)

### Step 1: Install Dependencies
Open a terminal in this directory and install the required packages:
```bash
python -m pip install -r requirements.txt
```

### Step 2: Launch the Dashboard
Run the Streamlit application:
```bash
python -m streamlit run streamlit_app.py
```

### Step 3: Access the Interface
Open your web browser and navigate to the local address displayed in the terminal (typically `http://localhost:8501`).

---

## User Guide: Making Predictions

1. Open the **Predict Employee Attrition** section from the sidebar navigation.
2. Complete the employee profile parameters (organized across demographics, job involvement, compensation, and satisfaction metrics).
3. Click **Predict Attrition Risk**.
4. The system outputs:
   * **Predicted Attrition Probability**: The statistical likelihood of departure (0.0% to 100.0%).
   * **Visual Progress Bar**: Instant risk visualization.
   * **Classification Banner**: HIGHER ATTRITION RISK if probability $\ge$ 0.39, or LOWER PREDICTED ATTRITION RISK if probability $<$ 0.39.
   * **Submitted Profile Audit**: Record of the submitted parameters for review.

---

## Ethical & Responsible AI Considerations

* **Decision Support Only**: Risk scores should guide constructive talent retention strategies and check-in conversations, never automated employment terminations or adverse actions.
* **Data Privacy**: Ensure individual employee assessments comply with organizational data governance and privacy policies.
