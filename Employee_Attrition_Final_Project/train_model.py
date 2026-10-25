"""Train and export the recall-focused employee attrition model used by Employee_Attrition_Modeling_Recall_Focused notebook."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = BASE_DIR / "attrition_model.joblib"
META_PATH = BASE_DIR / "model_metadata.json"
RANDOM_STATE = 42
TARGET_RECALL = 0.80


def main():
    df = pd.read_csv(DATA_PATH)
    data = df.dropna(subset=["Attrition"]).copy()
    y = data["Attrition"].map({"No": 0, "Yes": 1}).astype(int)
    X = data.drop(columns=["Attrition"])
    identifiers = [c for c in ["EmployeeNumber", "EmployeeID", "EmployeeId"] if c in X.columns]
    X = X.drop(columns=identifiers)
    constants = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constants)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (oof >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": float(precision_score(y_train, pred, zero_division=0)),
            "recall": float(recall_score(y_train, pred, zero_division=0)),
            "f1": float(f1_score(y_train, pred, zero_division=0)),
        })
    eligible = [r for r in rows if r["recall"] >= TARGET_RECALL]
    if eligible:
        chosen = sorted(eligible, key=lambda r: (r["precision"], r["f1"], r["threshold"]), reverse=True)[0]
        selection_note = f"Highest-precision threshold meeting cross-validation recall >= {TARGET_RECALL:.0%}."
    else:
        chosen = sorted(rows, key=lambda r: (r["recall"], r["precision"], r["f1"]), reverse=True)[0]
        selection_note = f"No tested threshold met {TARGET_RECALL:.0%} recall; selected highest cross-validation recall."

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= chosen["threshold"]).astype(int)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision_attrition_yes": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall_attrition_yes": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1_attrition_yes": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, prob)),
        "test_average_precision": float(average_precision_score(y_test, prob)),
    }
    bundle = {
        "model": model,
        "threshold": float(chosen["threshold"]),
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "category_options": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in categorical},
        "numeric_defaults": {c: float(X[c].median()) for c in numeric},
        "numeric_min": {c: float(X[c].min()) for c in numeric},
        "numeric_max": {c: float(X[c].max()) for c in numeric},
        "model_name": "Recall-focused Logistic Regression (class_weight='balanced')",
        "target_recall": TARGET_RECALL,
        "threshold_selection_note": selection_note,
    }
    joblib.dump(bundle, MODEL_PATH)
    metadata = {
        "dataset_rows": int(len(df)), "dataset_columns": int(df.shape[1]),
        "attrition_yes": int((df["Attrition"] == "Yes").sum()),
        "attrition_no": int((df["Attrition"] == "No").sum()),
        "training_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "chosen_threshold": float(chosen["threshold"]),
        "cross_validation_precision": chosen["precision"],
        "cross_validation_recall": chosen["recall"],
        "cross_validation_f1": chosen["f1"],
        "threshold_selection_note": selection_note,
        "test_metrics_at_selected_threshold": metrics,
        "excluded_identifier_columns": identifiers,
        "excluded_constant_columns": constants,
        "feature_count": int(X.shape[1]),
        "model_name": bundle["model_name"],
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
