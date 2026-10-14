import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

st.set_page_config(page_title="CreditScore AI", page_icon="📊", layout="wide")

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving credit utilization",
    "age": "Customer age (years)",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30–59 days past due",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines and loans",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberRealEstateLoansOrLines": "Real-estate loans / lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60–89 days past due",
    "NumberOfDependents": "Number of dependents",
}

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px;}
.hero {padding: 1.7rem 1.9rem; border-radius: 20px; background: linear-gradient(120deg,#101c3a,#1b4772); color: white; margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.2rem; margin: 0 0 .45rem 0; color: white;}
.hero p {color: #d7e8ff; font-size: 1rem; margin: 0;}
.metric-card {padding: 1rem 1.1rem; border: 1px solid #e4eaf2; border-radius: 14px; background: #ffffff;}
.small-muted {color: #64748b; font-size: .88rem;}
div[data-testid="stForm"] {border: 1px solid #e4eaf2; padding: 1.1rem; border-radius: 16px;}
</style>
<div class="hero">
  <h1>📊 Customer Credit Scoring</h1>
  <p>Risk assessment powered by XGBoost · Enter customer information to estimate serious delinquency risk over the next two years.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Model setup")
    st.write("Upload the **cs-training.csv** dataset used in your notebook to train the prediction model.")
    uploaded = st.file_uploader("Training dataset (CSV)", type=["csv"])
    st.caption("Expected target column: `SeriousDlqin2yrs`. The app trains the model in this session.")
    st.divider()
    st.markdown("**Models in the project**")
    st.write("• XGBoost — primary predictor\n• Logistic Regression — evaluated in the notebook")
    st.caption("Educational prototype only; not a lending decision system.")

@st.cache_resource(show_spinner=False)
def train_model(csv_bytes):
    from io import BytesIO
    df = pd.read_csv(BytesIO(csv_bytes))
    df = df.drop(columns=["Unnamed: 0"], errors="ignore").drop_duplicates()
    missing = [c for c in FEATURES + ["SeriousDlqin2yrs"] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    df = df[FEATURES + ["SeriousDlqin2yrs"]].copy()
    df["SeriousDlqin2yrs"] = pd.to_numeric(df["SeriousDlqin2yrs"], errors="coerce")
    df = df.dropna(subset=["SeriousDlqin2yrs"])
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["SeriousDlqin2yrs"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1-score": f1_score(y_test, preds, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probs),
        "Rows": len(df),
        "Positive rate": float(y.mean()),
    }
    return model, metrics

if uploaded is None:
    st.info("**Start here:** Upload `cs-training.csv` using the box in the left sidebar. The app needs the dataset to train the real model; predictions are not hard-coded.")
    st.subheader("What this app does")
    a, b, c = st.columns(3)
    with a:
        st.markdown("<div class='metric-card'><h3>01 · Train</h3><p>Fit an XGBoost model from the project dataset with median imputation inside the pipeline.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='metric-card'><h3>02 · Enter</h3><p>Provide the ten financial and repayment features used by your notebook.</p></div>", unsafe_allow_html=True)
    with c:
        st.markdown("<div class='metric-card'><h3>03 · Assess</h3><p>View the estimated probability and the model's threshold-based risk category.</p></div>", unsafe_allow_html=True)
    st.stop()

try:
    with st.spinner("Training XGBoost model on the uploaded dataset…"):
        model, metrics = train_model(uploaded.getvalue())
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

st.success(f"Model trained successfully on {metrics['Rows']:,} cleaned rows.")
st.subheader("Model validation snapshot")
cols = st.columns(5)
for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
    col.metric(key, f"{metrics[key]:.1%}")
st.caption(f"Positive target rate in uploaded data: {metrics['Positive rate']:.2%}. Metrics use a 0.50 probability threshold; the data is imbalanced, so recall and precision should be considered alongside accuracy.")

st.divider()
st.subheader("Customer information")
st.write("Enter values in the same units and format as the training dataset. If a value is unknown, leave it blank for median imputation.")
with st.form("customer_form"):
    c1, c2 = st.columns(2)
    values = {}
    defaults = {
        "RevolvingUtilizationOfUnsecuredLines": 0.35,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.30,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0,
    }
    integer_features = {"age", "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"}
    for i, feature in enumerate(FEATURES):
        col = c1 if i % 2 == 0 else c2
        with col:
            if feature == "MonthlyIncome":
                raw = st.text_input(LABELS[feature], value=str(defaults[feature]), help="Use the same currency units as the training dataset.")
            elif feature in integer_features:
                raw = st.number_input(LABELS[feature], min_value=0, max_value=120, value=int(defaults[feature]), step=1)
            else:
                raw = st.number_input(LABELS[feature], min_value=0.0, value=float(defaults[feature]), step=0.05, format="%.4f", help="Use the same definition as in the dataset.")
            values[feature] = raw
    submitted = st.form_submit_button("Assess credit risk", type="primary", use_container_width=True)

if submitted:
    try:
        row = {}
        for feature in FEATURES:
            v = values[feature]
            if feature == "MonthlyIncome" and str(v).strip() == "":
                row[feature] = np.nan
            else:
                row[feature] = float(v)
        customer = pd.DataFrame([row], columns=FEATURES)
        probability = float(model.predict_proba(customer)[0, 1])
        prediction = int(probability >= 0.5)
        st.divider()
        st.subheader("Prediction result")
        r1, r2 = st.columns([1, 1])
        with r1:
            st.metric("Estimated delinquency probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
        with r2:
            if prediction == 1:
                st.error("Higher predicted risk at the 0.50 threshold")
                st.write("The model predicts the positive target class: serious delinquency within two years.")
            else:
                st.success("Lower predicted risk at the 0.50 threshold")
                st.write("The model predicts the negative target class at the selected threshold.")
        st.caption("This is a model estimate, not a guaranteed outcome or a credit approval/denial. The default threshold is 0.50 and may not be optimal for imbalanced data.")
        with st.expander("Show submitted feature values"):
            st.dataframe(customer.rename(columns=LABELS), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not generate prediction: {e}")

st.divider()
st.caption("Customer Credit Scoring Using XGBoost and Logistic Regression · Academic project prototype")
