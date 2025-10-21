import streamlit as st
import pandas as pd
from io import StringIO

from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR


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
            df = pd.read_csv(sample_path)
            st.session_state["input_df"] = df
            st.success("Loaded sample data")
        except Exception as e:
            st.error(f"Failed to load sample data: {e}")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["input_df"] = df
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

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

                input_df = st.session_state["input_df"].copy()

                try:
                    y_pred = model.predict(input_df)
                except Exception:
                    y_pred = model.predict(input_df.values)

                input_df["predicted_column"] = y_pred
                try:
                    input_df["predicted_column"] = input_df["predicted_column"].replace(
                        TargetValueMapping().reverse_mapping()
                    )
                except Exception:
                    pass

                st.subheader("Predictions")
                st.dataframe(input_df.head())

                csv = input_df.to_csv(index=False)
                st.download_button("Download predictions CSV", csv, file_name="predictions.csv")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("Upload a CSV file or click 'Use sample data' to start")
