# Employee Attrition Prediction System

An AI-powered HR analytics dashboard for employee attrition risk assessment.

## Run the Dashboard Locally

1. Open a terminal in `Streamlit_Dashboard/`.
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Start the dashboard application:
   ```bash
   python -m streamlit run streamlit_app.py
   ```
4. Open the local address printed by Streamlit in your browser, normally `http://localhost:8501`.

---

## System Architecture & Pipeline Notes

- **Benchmark Dataset**: The project uses the IBM HR Analytics Employee Attrition dataset (1,470 records, 35 features) for workforce analytics and predictive modeling.
- **Unified Analytics Dashboard**: Includes workforce overview KPIs, multidimensional attrition analysis by department, overtime, and satisfaction factors, an interactive 30-feature risk prediction form, and comprehensive model performance metrics.
- **Machine Learning Pipeline**: A recall-focused Logistic Regression model with balanced class weighting, coupled with a median/StandardScaler numerical pipeline and a most-frequent/OneHotEncoder categorical pipeline.
- **Calibrated Decision Threshold**: Set at `0.39` to prioritize recall (>80%), minimizing false negatives in identifying at-risk talent.
- **Data Compatibility**: Uploading an alternate CSV dynamically updates the exploratory analytics; risk predictions utilize the standardized 30-feature schema.
- **Ethical Decision Support**: Predictions serve as decision-support guidance for proactive retention discussions rather than automated HR decisions.

---

## Submission Checklist

- [x] Streamlit dashboard loads trained model pipeline and metadata cleanly.
- [x] All 4 navigation sections (Overview, Attrition Analysis, Predict Employee Attrition, Model Performance) function without error.
- [x] Real-time risk prediction form tested and operational with probability scoring and calibrated thresholding.
- [x] Model performance metrics verified against held-out validation results.
