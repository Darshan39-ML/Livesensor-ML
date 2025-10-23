import os
import sys
import pandas as pd
import numpy as np

from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR


def main():
    try:
        resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not resolver.is_model_exists():
            print(f"No model found in {SAVED_MODEL_DIR}")
            sys.exit(2)

        model_path = resolver.get_best_model_path()
        print(f"Loading model from: {model_path}")
        model = load_object(file_path=model_path)
        print("Loaded model type:", type(model))

        # Choose a sample CSV to predict on
        sample_csv = "artifact/09_30_2025_21_13_38/data_ingestion/feature_store/sensor.csv"
        if not os.path.exists(sample_csv):
            # fallback: ingested test/train
            alt = os.path.join("artifact", "09_30_2025_21_13_38", "data_ingestion", "ingested", "test.csv")
            if os.path.exists(alt):
                sample_csv = alt
            else:
                print("No sample CSV found to run predictions on.")
                sys.exit(3)

        df = pd.read_csv(sample_csv)
        print(f"Loaded sample CSV: {sample_csv} rows={len(df)} columns={list(df.columns)[:10]}...")

        if "class" in df.columns:
            y_true = df["class"].astype(str).reset_index(drop=True)
            X = df.drop(columns=["class"])
        else:
            y_true = None
            X = df

        # Run prediction
        try:
            y_pred = model.predict(X)
        except Exception as e:
            print("model.predict(X) failed, trying model.predict(X.values). Error:", e)
            y_pred = model.predict(X.values)

        y_pred = np.asarray(y_pred).ravel()

        # Map numeric preds back to label strings if needed
        mapped_preds = None
        try:
            # if predictions are numeric (0/1), map to 'neg'/'pos'
            if np.issubdtype(y_pred.dtype, np.integer) or np.issubdtype(y_pred.dtype, np.floating):
                rev = TargetValueMapping().reverse_mapping()
                mapped_preds = [rev.get(int(v), v) for v in y_pred]
            else:
                mapped_preds = [str(x) for x in y_pred]
        except Exception:
            mapped_preds = [str(x) for x in y_pred]

        # Show sample of predictions
        sample_n = min(10, len(mapped_preds))
        print("Sample predictions (predicted -> true if available):")
        for i in range(sample_n):
            true = y_true.iloc[i] if y_true is not None else "<no-true>"
            print(f"  {mapped_preds[i]}  <--  {true}")

        if y_true is not None:
            y_true_list = y_true.tolist()
            # normalize both for comparison
            y_true_norm = [str(x).strip() for x in y_true_list]
            pred_norm = [str(x).strip() for x in mapped_preds]
            correct = sum(1 for a, b in zip(pred_norm, y_true_norm) if a == b)
            acc = correct / len(y_true_norm)
            print(f"Accuracy on sample CSV: {acc:.4f} ({correct}/{len(y_true_norm)})")
        else:
            print("No true labels present in CSV; can't compute accuracy.")

        # Exit 0 on completion
        return 0
    except Exception as e:
        print("ERROR during prediction run:", e)
        return 5


if __name__ == "__main__":
    sys.exit(main())
