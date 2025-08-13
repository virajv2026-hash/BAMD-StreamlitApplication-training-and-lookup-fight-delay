#pip install streamlit pandas scikit-learn matplotlib joblib
# (optional) pip install xgboost lightgbm catboost
#streamlit run (filename).py

# flight_delay_app.py
# One Streamlit app that:
# 1) Trains/benchmarks multiple models on your Airlines-style CSV
# 2) Saves the best model as delay_model.joblib automatically
# 3) Lets you look up a flight by number and predict using the saved or in-session model

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

# ---------- Page Setup ----------
st.set_page_config(page_title="✈️ Flight Delay — Train & Lookup", layout="wide")
st.title("✈️ Flight Delay — Train, Benchmark, and Lookup (One App)")

# ---------- Optional Models ----------
def optional_models():
    models = {}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            tree_method="hist", eval_metric="logloss"
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            depth=6, learning_rate=0.1, iterations=300,
            verbose=False, allow_writing_files=False
        )
    except Exception:
        pass
    return models


# ---------- Sidebar: Data (with sampling control) ----------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    default_path = "Airlines.csv"
    st.caption("If no file is uploaded, the app will try to read **Airlines.csv** from the current folder.")

    st.divider()
    st.subheader("Row Limit")
    rows_mode = st.radio("Rows to use", ["All rows", "Sample rows"], index=0, horizontal=True)
    max_rows = None
    sample_seed = None
    if rows_mode == "Sample rows":
        max_rows = st.number_input("Sample up to N rows", min_value=1000, max_value=1_000_000, value=20_000, step=1000)
        sample_seed = st.number_input("Sampling seed", value=42, step=1)
    st.caption("Sampling applies after loading and basic cleaning. Sampling is uniform random.")

# ---------- Data Load ----------
@st.cache_data
def load_csv(file_or_path: object) -> pd.DataFrame:
    """Load CSV either from an uploaded file-like object or a filesystem path.
       Normalizes column names to lowercase + trimmed."""
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.lower().str.strip()
    return df

df = None
try:
    if uploaded is not None:
        df = load_csv(uploaded)
    else:
        df = load_csv(default_path)
except Exception as e:
    st.warning(f"Could not load data: {e}")

if df is None or df.empty:
    st.error("No data loaded. Upload a CSV or place Airlines.csv in the working directory.")
    st.stop()

# Apply sampling if requested
if rows_mode == "Sample rows" and max_rows is not None and len(df) > int(max_rows):
    df = df.sample(int(max_rows), random_state=int(sample_seed) if sample_seed is not None else None)
    st.info(f"Sampled **{len(df):,}** rows for training/evaluation.")

st.success(f"Loaded data with shape {df.shape}")
st.dataframe(df.head(20), use_container_width=True)


# ---------- Helpers ----------
def guess_target_column(columns) -> str | None:
    candidates = ["delay", "delayed", "is_delay", "is_delayed", "target", "label"]
    for c in candidates:
        if c in columns:
            return c
    return None

def guess_flight_id_column(columns) -> str | None:
    candidates = ["flight_number", "flightno", "flight_no", "flightid", "flight", "flightcode"]
    for c in candidates:
        if c in columns:
            return c
    return None

def make_onehot_encoder():
    """Handle sklearn version differences (sparse vs sparse_output)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # older sklearn

def build_preprocessor(X: pd.DataFrame):
    # Treat object dtype and very low-cardinality integers as categorical
    categorical_cols = [
        c for c in X.columns
        if (X[c].dtype == "object") or (pd.api.types.is_integer_dtype(X[c]) and X[c].nunique() <= 30)
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    pre = ColumnTransformer(
        transformers=[
            ("cat", make_onehot_encoder(), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )
    return pre, categorical_cols, numeric_cols

# ---------- Sidebar: Schema ----------
with st.sidebar:
    st.header("Schema")
    target_col_choice = st.selectbox("Target (label) column",
                                     options=["<detect>"] + list(df.columns),
                                     index=0)
    if target_col_choice == "<detect>":
        target_col = guess_target_column(df.columns)
        if target_col is None:
            st.error("Target column not detected. Please select one.")
            st.stop()
        st.caption(f"Detected target column: **{target_col}**")
    else:
        target_col = target_col_choice

    flight_id_choice = st.selectbox("Flight number column (for lookup)",
                                    options=["<detect>"] + list(df.columns),
                                    index=0)
    if flight_id_choice == "<detect>":
        flight_id_col = guess_flight_id_column(df.columns)
        if flight_id_col is None:
            st.warning("Flight number column not detected. You can still choose it below.")
        else:
            st.caption(f"Detected flight id column: **{flight_id_col}**")
    else:
        flight_id_col = flight_id_choice

# ---------- Mode Switch ----------
mode = st.radio("Mode", ["Train & Benchmark", "Lookup / Predict"], horizontal=True)

# ---------- Models ----------
base_models = {
    "LogReg": LogisticRegression(solver="liblinear", max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "GaussianNB": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42),
}
base_models.update(optional_models())

# ---------- Train & Benchmark ----------
if mode == "Train & Benchmark":
    st.subheader("Model Training & Comparison")

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in the data.")
        st.stop()

    drop_cols = [target_col]
    if flight_id_col in df.columns:
        drop_cols.append(flight_id_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    # If labels look binary {0,1}, coerce to int
    uniq = pd.Series(y).dropna().unique()
    if set(uniq).issubset({0, 1}):
        y = y.astype(int)

    pre, cat_cols, num_cols = build_preprocessor(X)

    with st.sidebar:
        st.header("Training")
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random seed", value=42, step=1)
        selected_names = st.multiselect("Models", options=list(base_models.keys()),
                                        default=list(base_models.keys())[:3])
        metric_choice = st.selectbox("Primary metric", ["ROC-AUC", "F1", "Accuracy"], index=0)

    stratify_y = y if len(pd.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    if st.button("Run Benchmark"):
        results = []

        for name in selected_names:
            est = base_models[name]
            pipe = Pipeline(steps=[("pre", pre), ("model", est)])

            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                roc_auc = np.nan
                try:
                    if len(np.unique(y)) == 2:
                        if hasattr(pipe, "predict_proba"):
                            probs = pipe.predict_proba(X_test)[:, 1]
                            roc_auc = roc_auc_score(y_test, probs)
                        elif hasattr(pipe, "decision_function"):
                            scores = pipe.decision_function(X_test)
                            roc_auc = roc_auc_score(y_test, scores)
                    else:
                        if hasattr(pipe, "predict_proba"):
                            probs = pipe.predict_proba(X_test)
                            roc_auc = roc_auc_score(pd.get_dummies(y_test), probs, multi_class="ovr")
                except Exception:
                    pass

                results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC-AUC": roc_auc})

                with st.expander(f"Details: {name}", expanded=False):
                    st.write("**Classification Report**")
                    st.text(classification_report(y_test, y_pred))

                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    st.dataframe(pd.DataFrame(cm), use_container_width=True)

                    if len(np.unique(y)) == 2:
                        try:
                            fig = plt.figure()
                            RocCurveDisplay.from_estimator(pipe, X_test, y_test)
                            st.pyplot(fig)
                        except Exception:
                            st.info("ROC curve not available for this model.")
            except Exception as e:
                st.warning(f"{name} failed to train: {e}")
                continue

        if results:
            res_df = pd.DataFrame(results).sort_values(by=metric_choice, ascending=False)
            st.subheader("Results")
            st.dataframe(res_df, use_container_width=True)

            best_name = res_df.iloc[0]["Model"]
            st.success(f"Best by **{metric_choice}**: **{best_name}**")

            # Refit best on ALL data, save to session and disk, and offer download
            best_est = base_models[best_name]
            best_pipe = Pipeline(steps=[("pre", pre), ("model", best_est)])
            best_pipe.fit(X, y)

            st.session_state["best_pipe"] = best_pipe
            st.session_state["schema"] = {
                "target_col": target_col,
                "flight_id_col": flight_id_col if flight_id_col in df.columns else None,
                "feature_columns": list(X.columns),
            }

            # Save to disk by default
            joblib.dump({"pipeline": best_pipe, "schema": st.session_state["schema"]}, "delay_model.joblib")
            st.success("✅ Best model saved as **delay_model.joblib** in the working directory.")

            # Also provide download directly
            buf = BytesIO()
            joblib.dump({"pipeline": best_pipe, "schema": st.session_state["schema"]}, buf)
            st.download_button(
                "⬇️ Download trained model (.joblib)",
                data=buf.getvalue(),
                file_name="delay_model.joblib",
                mime="application/octet-stream",
            )

# ---------- Lookup / Predict ----------
if mode == "Lookup / Predict":
    st.subheader("Lookup by Flight Number (Exact Match) and Predict")

    # First try session model
    model = st.session_state.get("best_pipe", None)
    schema = st.session_state.get("schema", None)

    # If not in session, try auto-loading from disk
    if model is None:
        try:
            obj = joblib.load("delay_model.joblib")
            if isinstance(obj, dict) and "pipeline" in obj:
                model = obj["pipeline"]
                schema = obj.get("schema", schema)
            else:
                model = obj
            st.info("Loaded model from **delay_model.joblib**.")
        except FileNotFoundError:
            st.warning("No trained model found in session or disk. Train first, or upload a .joblib file below.")

    # Upload can override
    model_file = st.file_uploader("Upload a saved model (.joblib) (optional)", type=["joblib"], key="model_uploader")
    if model_file is not None:
        try:
            obj = joblib.load(model_file)
            if isinstance(obj, dict) and "pipeline" in obj:
                model = obj["pipeline"]
                schema = obj.get("schema", schema)
            else:
                model = obj
            st.success("Model loaded from uploaded file.")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

    # If no schema yet, infer a minimal one
    if schema is None:
        schema = {
            "target_col": target_col if target_col in df.columns else None,
            "flight_id_col": flight_id_col if flight_id_col in df.columns else None,
            "feature_columns": [c for c in df.columns if c not in [target_col, flight_id_col]],
        }

    # Verify flight id column exists
    if not schema.get("flight_id_col") or schema["flight_id_col"] not in df.columns:
        st.error("Please set a valid flight number column in the sidebar (Schema).")
        st.stop()

    # Input for flight number — exact match, case-insensitive, trimmed
    flight_no = st.text_input("Enter Flight Number (exact match)").strip()
    if flight_no:
        mask = df[schema["flight_id_col"]].astype(str).str.strip().str.lower() == flight_no.lower()
        matches = df[mask]

        if matches.empty:
            st.warning("Flight not found.")
        else:
            row = matches.iloc[0].copy()
            st.write("**Matched Row:**")
            st.dataframe(row.to_frame().T, use_container_width=True)

            # Predict if a model is available; else fall back to dataset label
            if model is not None:
                feats = [c for c in schema.get("feature_columns", []) if c in df.columns]
                X_row = row[feats].to_frame().T
                try:
                    pred = model.predict(X_row)[0]
                    prob_text = ""
                    try:
                        proba = model.predict_proba(X_row)
                        if proba.shape[1] == 2:
                            prob_text = f" (prob. delayed: {proba[0,1]:.2%})"
                    except Exception:
                        pass
                    if int(pred) == 1:
                        st.error(f"Model Prediction: DELAYED{prob_text}")
                    else:
                        st.success(f"Model Prediction: ON TIME{prob_text}")
                except Exception as e:
                    st.warning(f"Prediction failed: {e}")
            else:
                tgt = schema.get("target_col")
                if tgt and tgt in row.index:
                    try:
                        lab = int(row[tgt])
                    except Exception:
                        lab = row[tgt]
                    if lab == 1:
                        st.error("Label in data: DELAYED")
                    elif lab == 0:
                        st.success("Label in data: ON TIME")
                    else:
                        st.info(f"Label in data: {lab}")
                else:
                    st.info("No model loaded and no target label present; cannot determine delay status.")
