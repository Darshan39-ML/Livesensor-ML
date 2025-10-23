import streamlit as st
import pandas as pd
from io import StringIO

from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
import traceback
import os
import numpy as np


st.set_page_config(page_title="Livesensor-ML Demo", layout="wide")

st.title("Livesensor-ML — Prediction Demo")

st.markdown(
    "Upload a CSV with the same feature columns used during training and click Predict to see model outputs."
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

col1, col2 = st.columns([3, 1])

with col2:
    if st.button("Use sample data"):
        try:
            sample_path = "artifact/09_30_2025_21_13_38/data_ingestion/feature_store/sensor.csv"
            # Validate file exists and is readable before attempting to load
            if not os.path.exists(sample_path):
                st.error(f"Sample file not found: {sample_path}")
                st.info("Ensure the sample file exists in the repository under 'artifact/...'.")
            elif not os.access(sample_path, os.R_OK):
                st.error(f"Sample file exists but is not readable (permissions): {sample_path}")
                try:
                    st.text(str(os.stat(sample_path)))
                except Exception:
                    pass
            else:
                df = pd.read_csv(sample_path)
                st.session_state["input_df"] = df
                st.success("Loaded sample data")
        except Exception as e:
            tb = traceback.format_exc()
            st.error("Failed to load sample data. See details in the expander below.")
            with st.expander("Sample load error details"):
                st.text(f"Exception: {e}")
                st.text(tb)

if uploaded_file is not None:
    try:
        try:
            # uploaded_file is a BytesIO-like object provided by Streamlit
            # Save a copy to workspace to make debugging easier and to ensure
            # the app has access to a concrete file path.
            uploaded_bytes = uploaded_file.getvalue()
            save_dir = os.path.join(os.getcwd(), ".uploaded_files")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_bytes)

            # Now attempt to read from the saved path as a CSV
            df = pd.read_csv(save_path)
            st.session_state["input_df"] = df
            st.success(f"Uploaded and loaded CSV: {uploaded_file.name}")
            st.info(f"Saved upload to: {save_path}")
        except Exception as e:
            tb = traceback.format_exc()
            st.error("Failed to read uploaded CSV. See details in the expander below.")
            with st.expander("Upload error details"):
                st.text(f"Exception: {e}")
                st.text(tb)
    except Exception as e:
        tb = traceback.format_exc()
        st.error("Failed to process uploaded file. See details:")
        with st.expander("Upload processing details"):
            st.text(f"Exception: {e}")
            st.text(tb)

if "input_df" in st.session_state:
    st.subheader("Input preview")
    st.dataframe(st.session_state["input_df"].head())

    model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
    if not model_resolver.is_model_exists():
        st.warning("No trained model found (SAVED_MODEL_DIR). Upload or run training first.")
    else:
        if st.button("Predict"):
            try:
                best_model_path = model_resolver.get_best_model_path()
                model = load_object(file_path=best_model_path)

                # prepare input dataframe and align to model expected features
                input_df = st.session_state["input_df"].copy()

                # determine expected features from the model's preprocessor
                expected = None
                try:
                    pre = getattr(model, 'preprocessor', None)
                    if pre is not None and hasattr(pre, 'feature_names_in_'):
                        expected = list(pre.feature_names_in_)
                    else:
                        # inspect pipeline steps for feature_names_in_
                        if pre is not None and hasattr(pre, 'named_steps'):
                            for step in pre.named_steps.values():
                                if hasattr(step, 'feature_names_in_'):
                                    expected = list(step.feature_names_in_)
                                    break
                except Exception:
                    expected = None

                if expected is not None and hasattr(input_df, 'columns'):
                    # add missing columns with NA
                    missing = [c for c in expected if c not in input_df.columns]
                    extra = [c for c in input_df.columns if c not in expected]
                    if missing:
                        for c in missing:
                            input_df[c] = pd.NA
                    # Replace pandas NA with np.nan for sklearn compatibility
                    input_df = input_df.replace({pd.NA: np.nan})
                    # coerce columns to numeric where possible
                    for c in input_df.columns:
                        input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
                    if extra:
                        input_df = input_df.drop(columns=extra)
                    # reorder
                    input_df = input_df[expected]
                    if missing or extra:
                        st.info(f"Aligned input columns: added {len(missing)} missing, dropped {len(extra)} extra columns")

                try:
                    y_pred = model.predict(input_df)
                except Exception:
                    # as a last resort try numpy values
                    y_pred = model.predict(input_df.values)

                input_df["predicted_column"] = y_pred

                # compute prediction confidence (percentage) when possible
                confidence_scores = None
                try:
                    # Try wrapper-level predict_proba first
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_df)
                    else:
                        # try underlying estimator
                        est = getattr(model, 'model', model)
                        pre = getattr(model, 'preprocessor', None)
                        if hasattr(est, 'predict_proba'):
                            X_for_proba = input_df
                            if pre is not None:
                                X_for_proba = pre.transform(input_df)
                            proba = est.predict_proba(X_for_proba)
                        else:
                            proba = None
                    if proba is not None:
                        # take the max probability across classes as confidence
                        import numpy as _np
                        confidence_scores = _np.max(proba, axis=1)
                except Exception:
                    confidence_scores = None

                if confidence_scores is not None:
                    input_df["prediction_confidence_score"] = confidence_scores
                    input_df["prediction_confidence"] = [f"{(s*100):.1f}%" for s in confidence_scores]
                else:
                    input_df["prediction_confidence_score"] = ""
                    input_df["prediction_confidence"] = ""

                try:
                    input_df["predicted_column"] = input_df["predicted_column"].replace(
                        TargetValueMapping().reverse_mapping()
                    )
                except Exception:
                    pass

                st.subheader("Predictions")

                # Show overall summary: class distribution and average confidence
                try:
                    summary_cols = []
                    pct_series = input_df['predicted_column'].value_counts(normalize=True)
                    pct_text = " — ".join([f"{label}: {(frac*100):.1f}%" for label, frac in pct_series.items()])
                    avg_conf_text = ""
                    if 'prediction_confidence_score' in input_df.columns and input_df['prediction_confidence_score'].dtype != object:
                        avg_conf = float(input_df['prediction_confidence_score'].mean())
                        avg_conf_text = f"Average confidence: {(avg_conf*100):.1f}%"

                    st.markdown(f"**Summary:** {pct_text} {('|' + avg_conf_text) if avg_conf_text else ''}")
                except Exception:
                    pass

                st.dataframe(input_df.head())

                csv = input_df.to_csv(index=False)
                st.download_button("Download predictions CSV", csv, file_name="predictions.csv")

            except Exception as e:
                # Provide a clearer, actionable message for common model loading
                # and XGBoost artifact incompatibility failures.
                err_msg = str(e)
                tb = traceback.format_exc()

                # Heuristic: look for xgboost/booster/handle related errors
                lowered = err_msg.lower() + "\n" + tb.lower()
                is_xgb_incompat = any(k in lowered for k in ["xgboost", "booster", "handle", "artifact incompat", "invalid cast"]) 

                if is_xgb_incompat:
                    st.error("Model prediction failed due to an XGBoost artifact incompatibility or corrupted Booster state.")
                    with st.expander("Why this happened and how to fix it"):
                        st.markdown(
                            "The model file appears to contain XGBoost binary state that cannot be rehydrated in this environment. "
                            "This commonly happens when a Booster was pickled or saved with a different xgboost/libxgboost runtime than the one currently installed."
                        )
                        st.markdown("Recommended remediation steps:")
                        st.markdown(
                            "1. Re-export the model from the training environment using a portable format: `booster.save_model('model.json')` or `booster.save_model('model.bin')`.\n"
                            "2. If you used an sklearn wrapper (XGBClassifier/XGBRegressor), re-save a sklearn-compatible artifact using the same xgboost version as your runtime, or export the underlying booster as JSON/BIN.\n"
                            "3. Place the new artifact into the `saved_models/<timestamp>/` directory (the app uses `SAVED_MODEL_DIR`) and retry."
                        )
                        st.markdown("If you can't re-export the model, consider retraining or deploying the app in the original training environment (matching xgboost).")
                        st.markdown("Model file:")
                        st.code(best_model_path if 'best_model_path' in locals() else "(unknown)")
                        st.markdown("Full error (expand to inspect):")
                        st.text(tb)
                else:
                    st.error(f"Prediction failed: {err_msg}")
                    with st.expander("Details"):
                        st.text(tb)
else:
    st.info("Upload a CSV file or click 'Use sample data' to start")
