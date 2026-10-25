from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Backward compatibility shim for unpickling ColumnTransformer from scikit-learn <= 1.6
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, '_RemainderColsList'):
        from collections import UserList
        class _RemainderColsList(UserList):
            def __init__(self, initlist=None, future_dtype=None, warning_was_emitted=False, warning_enabled=True):
                super().__init__(initlist)
                self.future_dtype = future_dtype
                self.warning_was_emitted = warning_was_emitted
                self.warning_enabled = warning_enabled
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent

# Flexible path resolution for root deployment or subfolder deployment
candidate_data_paths = [
    BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    BASE_DIR / "member 3_dashboard" / "Streamlit_Dashboard" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    BASE_DIR / "DS MAJOR PROJECT" / "PROJECT" / "member 3_dashboard" / "Streamlit_Dashboard" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
]
DATA_PATH = next((p for p in candidate_data_paths if p.exists()), candidate_data_paths[0])

candidate_model_dirs = [
    BASE_DIR / "M2_Model_Files_For_M3",
    BASE_DIR / "member 3_dashboard" / "Streamlit_Dashboard" / "M2_Model_Files_For_M3",
    BASE_DIR / "DS MAJOR PROJECT" / "PROJECT" / "member 3_dashboard" / "Streamlit_Dashboard" / "M2_Model_Files_For_M3",
]
MODEL_DIR = next((p for p in candidate_model_dirs if p.exists()), candidate_model_dirs[0])
MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
META_PATH = MODEL_DIR / "model_metadata.json"
M2_DIR = MODEL_DIR  # Internal alias

st.set_page_config(page_title="Employee Attrition Prediction System", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

# Polished, professional styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; background: linear-gradient(125deg, #102a45 0%, #1a4971 100%); color: white; margin-bottom: 1.2rem; box-shadow: 0 4px 12px rgba(16,42,69,0.12);}
.hero h1 {font-size: 2.1rem; font-weight: 700; margin: 0 0 .35rem 0; color: #ffffff; letter-spacing: -0.5px;}
.hero p {margin: 0; color: #dbeefd; font-size: 1.05rem; font-weight: 400;}
.metric-note {color: #667085; font-size: .82rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.18); background: rgba(255,255,255,0.02); padding: 14px 16px; border-radius: 12px;}
.section-subtitle {color: #556987; margin-top: -0.5rem; margin-bottom: 1.1rem; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        return None
    try:
        return joblib.load(PREPROCESSOR_PATH)
    except Exception:
        return None

@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Internal compatibility aliases
load_m2_model = load_trained_model
load_m2_preprocessor = load_preprocessor
load_m2_metadata = load_metadata

def get_model_features(model, meta, default_df):
    feature_columns = meta.get("input_feature_columns", [])
    cat_cols = []
    num_cols = []
    category_options = {}

    preprocessor = None
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]

    if preprocessor and hasattr(preprocessor, "transformers_"):
        for name, trans, cols in preprocessor.transformers_:
            if name == "categorical":
                cat_cols = list(cols)
                if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
                    ohe = trans.named_steps["onehot"]
                    if hasattr(ohe, "categories_"):
                        for col, cats in zip(cols, ohe.categories_):
                            category_options[col] = list(cats)
            elif name == "numeric":
                num_cols = list(cols)

    if not cat_cols:
        cat_cols = [c for c in feature_columns if c in default_df.columns and default_df[c].dtype == "object"]
    if not num_cols:
        num_cols = [c for c in feature_columns if c not in cat_cols]

    for c in cat_cols:
        if c not in category_options:
            if c in default_df.columns:
                category_options[c] = sorted(default_df[c].dropna().astype(str).unique().tolist())
            else:
                category_options[c] = [""]

    numeric_min = {}
    numeric_max = {}
    numeric_defaults = {}
    for c in num_cols:
        if c in default_df.columns and pd.api.types.is_numeric_dtype(default_df[c]):
            numeric_min[c] = int(default_df[c].min())
            numeric_max[c] = int(default_df[c].max())
            numeric_defaults[c] = int(round(default_df[c].median()))
        else:
            numeric_min[c] = 0
            numeric_max[c] = 100
            numeric_defaults[c] = 10

    if not feature_columns:
        feature_columns = [c for c in default_df.columns if c not in ["Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]]

    return {
        "feature_columns": feature_columns,
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "category_options": category_options,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "numeric_defaults": numeric_defaults,
    }

def predict_attrition(model, preprocessor, input_df):
    has_preprocessor = hasattr(model, "named_steps") and "preprocessor" in model.named_steps
    if has_preprocessor or preprocessor is None:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_df)[0, 1])
        else:
            prob = float(model.predict(input_df)[0])
    else:
        transformed = preprocessor.transform(input_df)
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(transformed)[0, 1])
        else:
            prob = float(model.predict(transformed)[0])
    return prob

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)

def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def percent(value):
    return f"{value:.1f}%"

def rate_table(df, column):
    if column not in df.columns or "Attrition" not in df.columns:
        return pd.DataFrame()
    tmp = df[[column, "Attrition"]].copy()
    tmp[column] = tmp[column].fillna("Unknown").astype(str)
    grouped = tmp.groupby(column, dropna=False)["Attrition"].agg(
        employees="size", attrition_yes=lambda s: (s.astype(str).str.lower() == "yes").sum()
    ).reset_index()
    grouped["attrition_rate"] = np.where(grouped["employees"] > 0, 100 * grouped["attrition_yes"] / grouped["employees"], 0)
    return grouped.sort_values("attrition_rate", ascending=False)

def bar_chart(data, x, y, title, ylabel, rotate=0):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(data[x].astype(str), data[y])
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=rotate)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.sidebar.title("🏢 Attrition Analytics")
st.sidebar.caption("AI-Powered HR Risk Assessment & Workforce Intelligence System")
st.sidebar.divider()
uploaded = st.sidebar.file_uploader("Upload employee dataset (CSV)", type=["csv"], help="Upload a dataset with an Attrition column and compatible HR fields to explore custom workforce data.")
if uploaded is not None:
    try:
        df = clean_column_names(pd.read_csv(uploaded))
        source_label = f"Uploaded file: {uploaded.name}"
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")
        df = load_default_data()
        source_label = "IBM HR Analytics Benchmark Dataset"
else:
    df = load_default_data()
    source_label = "IBM HR Analytics Benchmark Dataset"

if "Attrition" in df.columns:
    df["Attrition"] = df["Attrition"].astype(str).str.strip().str.title()

st.markdown("""
<div class="hero">
  <h1>Employee Attrition Prediction System</h1>
  <p>An AI-powered HR analytics dashboard for employee attrition risk assessment.</p>
</div>
""", unsafe_allow_html=True)
st.caption(f"Data source: {source_label} · {len(df):,} records · {df.shape[1]} columns")

if "Attrition" not in df.columns:
    st.error("The selected dataset does not contain an 'Attrition' column. Upload the IBM HR Analytics dataset or a compatible CSV.")
    st.stop()

page = st.sidebar.radio("Navigation", ["Overview", "Attrition Analysis", "Predict Employee Attrition", "Model Performance"], index=0)

# Common data calculations
attrition_yes = (df["Attrition"].astype(str).str.lower() == "yes").sum()
attrition_no = (df["Attrition"].astype(str).str.lower() == "no").sum()
attrition_rate = 100 * attrition_yes / len(df) if len(df) else 0

if page == "Overview":
    st.subheader("Workforce overview")
    st.markdown('<p class="section-subtitle">A quick summary of employee retention and dataset quality.</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total employees", f"{len(df):,}")
    c2.metric("Employees left", f"{attrition_yes:,}")
    c3.metric("Employees stayed", f"{attrition_no:,}")
    c4.metric("Overall attrition", percent(attrition_rate))
    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Attrition distribution")
        dist = pd.DataFrame({"Status": ["Stayed", "Left"], "Employees": [attrition_no, attrition_yes]})
        bar_chart(dist, "Status", "Employees", "Employees by attrition status", "Employee count")
    with right:
        st.markdown("#### Dataset quality check")
        quality = pd.DataFrame({"Check": ["Missing values", "Duplicate rows", "Columns"], "Count": [int(df.isna().sum().sum()), int(df.duplicated().sum()), int(df.shape[1])]})
        st.dataframe(quality, hide_index=True, use_container_width=True)
        st.markdown("#### Dataset preview")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
    st.info("Attrition rates describe patterns in this dataset; they do not establish that a particular factor causes employees to leave.")

elif page == "Attrition Analysis":
    st.subheader("Attrition patterns")
    st.markdown('<p class="section-subtitle">Compare attrition rates across departments, overtime, job satisfaction, and work-life balance.</p>', unsafe_allow_html=True)
    dimensions = [
        ("Department", "Attrition by department"),
        ("OverTime", "Attrition by overtime status"),
        ("JobSatisfaction", "Attrition by job satisfaction"),
        ("WorkLifeBalance", "Attrition by work-life balance"),
    ]
    for row_start in (0, 2):
        cols = st.columns(2)
        for col, (field, title) in zip(cols, dimensions[row_start:row_start + 2]):
            with col:
                st.markdown(f"#### {title}")
                summary = rate_table(df, field)
                if summary.empty:
                    st.warning(f"Column '{field}' is not available in this dataset.")
                else:
                    bar_chart(summary, field, "attrition_rate", title, "Attrition rate (%)", 20 if field == "Department" else 0)
                    display = summary.rename(columns={field: "Group", "employees": "Employees", "attrition_yes": "Left", "attrition_rate": "Attrition rate (%)"}).copy()
                    display["Attrition rate (%)"] = display["Attrition rate (%)"].round(2)
                    st.dataframe(display, use_container_width=True, hide_index=True)
    with st.expander("Explore other columns"):
        available = [c for c in df.columns if c != "Attrition"]
        chosen = st.selectbox("Choose a category", available)
        summary = rate_table(df, chosen)
        if not summary.empty:
            st.dataframe(summary.rename(columns={chosen: "Group", "employees": "Employees", "attrition_yes": "Left", "attrition_rate": "Attrition rate (%)"}), use_container_width=True, hide_index=True)

elif page == "Predict Employee Attrition":
    st.subheader("Predict Employee Attrition Risk")
    st.markdown('<p class="section-subtitle">Enter an employee profile to estimate attrition risk probability using the trained machine learning pipeline. Output serves as decision support for proactive talent retention.</p>', unsafe_allow_html=True)
    
    model = load_trained_model()
    preprocessor = load_preprocessor()
    meta = load_metadata()
    
    if model is None:
        st.error("The trained model pipeline artifact could not be found. Please ensure the model file is available in the model directory.")
        st.stop()
        
    feat_spec = get_model_features(model, meta, df)
    features = feat_spec["feature_columns"]
    cat_cols = feat_spec["categorical_features"]
    num_cols = feat_spec["numeric_features"]
    threshold = float(meta.get("decision_threshold", 0.39))
    
    with st.form("prediction_form"):
        st.markdown("#### Employee Profile & Job Details")
        values = {}
        # Three-column input layout keeps the long form organized.
        col_sets = st.columns(3)
        for i, field in enumerate(features):
            with col_sets[i % 3]:
                if field in cat_cols:
                    options = feat_spec["category_options"].get(field, [])
                    if not options:
                        options = [""]
                    default = "No" if field == "OverTime" and "No" in options else options[0]
                    idx = options.index(default) if default in options else 0
                    values[field] = st.selectbox(field, options, index=idx, key=f"pred_{field}")
                else:
                    lo = int(feat_spec["numeric_min"].get(field, 0))
                    hi = int(feat_spec["numeric_max"].get(field, 100))
                    default = int(round(feat_spec["numeric_defaults"].get(field, lo)))
                    if hi < lo:
                        hi = lo
                    values[field] = st.number_input(field, min_value=lo, max_value=hi, value=min(max(default, lo), hi), step=1, key=f"pred_{field}")
        submitted = st.form_submit_button("Predict Attrition Risk", type="primary", use_container_width=True)
        
    if submitted:
        input_df = pd.DataFrame([{c: values[c] for c in features}], columns=features)
        probability = predict_attrition(model, preprocessor, input_df)
        predicted_yes = probability >= threshold
        
        left, right = st.columns([1, 2])
        with left:
            st.metric("Predicted Attrition Probability", percent(probability * 100))
            st.progress(min(max(probability, 0.0), 1.0))
        with right:
            if predicted_yes:
                st.error("Prediction Result: HIGHER ATTRITION RISK (Attrition = Yes)")
                st.write(f"The estimated probability ({percent(probability * 100)}) meets or exceeds the calibrated decision threshold ({threshold:.2f}).")
            else:
                st.success("Prediction Result: LOWER PREDICTED ATTRITION RISK (Attrition = No)")
                st.write(f"The estimated probability ({percent(probability * 100)}) is below the calibrated decision threshold ({threshold:.2f}).")
            st.caption(f"Decision threshold: {threshold:.2f} · Architecture: {meta.get('model_name', 'Recall-focused Logistic Regression')} · Probability is a statistical risk score for retention planning.")
        st.markdown("#### Submitted Employee Profile")
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        st.warning("Use predictions to support fair, constructive retention discussions. Do not use model predictions alone to make automated employment decisions.")

elif page == "Model Performance":
    st.subheader("Model Performance & Evaluation Methodology")
    st.markdown('<p class="section-subtitle">Comprehensive evaluation metrics, validation methodology, and pipeline architecture for the attrition prediction system.</p>', unsafe_allow_html=True)
    meta = load_metadata()
    model = load_trained_model()
    
    if meta:
        model_name = meta.get("model_name", "Recall-focused Logistic Regression")
        version = meta.get("model_version", "1.0")
        feat_count = meta.get("input_feature_count", len(meta.get("input_feature_columns", [])))
        threshold = float(meta.get("decision_threshold", 0.39))
        
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Selected Algorithm", model_name)
        m_c2.metric("Input Features", f"{feat_count} variables")
        m_c3.metric("Decision Threshold", f"{threshold:.2f}")
        
        rule = meta.get("prediction_rule", "")
        if rule:
            st.info(f"**Classification Decision Rule:** {rule}")
            
        st.markdown("#### Model Evaluation Results (Held-Out Test Set)")
        raw_metrics = meta.get("evaluation_metrics", {})
        metric_rows = []
        for metric_name, val in raw_metrics.items():
            if isinstance(val, (int, float)):
                metric_rows.append({"Metric": metric_name, "Score": f"{val:.4f}"})
            else:
                metric_rows.append({"Metric": metric_name, "Score": str(val)})
                
        if metric_rows:
            metric_df = pd.DataFrame(metric_rows)
            st.dataframe(metric_df, hide_index=True, use_container_width=True)
            
        notes = meta.get("notes", "")
        if notes:
            st.caption(f"**Pipeline Configuration:** {notes}")
    else:
        st.warning("Model metadata file was not found.")

    with st.expander("Model Architecture & Methodology Details"):
        st.markdown("""
        1. **Target Variable**: Predicts employee **Attrition** (`Yes` = 1 [at risk], `No` = 0 [retained]).
        2. **Feature Preprocessing Pipeline**: 
           - **Numeric Features (23)**: Median imputation followed by standard scaling (`StandardScaler`).
           - **Categorical Features (7)**: Most-frequent imputation followed by one-hot encoding (`OneHotEncoder(handle_unknown='ignore')`).
           - Integrated within a unified `ColumnTransformer` to eliminate training-serving data leakage.
        3. **Classification Algorithm**: `LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)`.
        4. **Recall-Focused Threshold Optimization**: 
           - In employee attrition risk modeling, the cost of a false negative (failing to identify an employee who leaves) substantially exceeds that of a false positive.
           - Cross-validation threshold optimization identified **0.39** as the optimal threshold, achieving **>80% recall** on the positive attrition class.
        5. **End-to-End Pipeline**: The model artifact encapsulates fitted preprocessing and classifier weights into a single reproducible pipeline.
        """)
    st.warning("The model was developed and validated on benchmark HR Analytics data. In a production environment, periodic retraining and fairness auditing are recommended to account for evolving workforce dynamics.")

st.sidebar.divider()
st.sidebar.caption("Employee Attrition Prediction System · Major Project")
