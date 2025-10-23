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
        model_path = resolver.get_best_model_path()
        print(f"Loading model from {model_path}")
        model = load_object(file_path=model_path)
        estimator = getattr(model, 'model', None)
        pre = getattr(model, 'preprocessor', None)
        print('Estimator type:', type(estimator))

        sample_csv = 'artifact/09_30_2025_21_13_38/data_ingestion/feature_store/sensor.csv'
        df = pd.read_csv(sample_csv)

        # Prepare X aligned to expected features
        expected = None
        if pre is not None:
            if hasattr(pre, 'feature_names_in_'):
                expected = list(pre.feature_names_in_)
            else:
                try:
                    if hasattr(pre, 'named_steps'):
                        for step in pre.named_steps.values():
                            if hasattr(step, 'feature_names_in_'):
                                expected = list(step.feature_names_in_)
                                break
                except Exception:
                    expected = None
        if expected is not None:
            X = df.copy()
            for c in expected:
                if c not in X.columns:
                    X[c] = pd.NA
            X = X[expected]
        else:
            X = df.drop(columns=['class']) if 'class' in df.columns else df

        # transform
        X_trans = pre.transform(X)

        # Apply compatibility shims for XGBoost sklearn wrappers
        if estimator is not None:
            try:
                cls_name = getattr(estimator, '__class__').__name__
                if cls_name.startswith('XGB'):
                    for attr, val in (
                        ('use_label_encoder', False),
                        ('gpu_id', None),
                        ('predictor', None),
                        ('n_jobs', 1),
                    ):
                        if not hasattr(estimator, attr):
                            try:
                                setattr(estimator, attr, val)
                                print(f"Set estimator.{attr} = {val}")
                            except Exception:
                                pass
            except Exception:
                pass

        # Try predict
        try:
            y_pred = estimator.predict(X_trans)
            print('predict succeeded via estimator.predict')
        except Exception as e:
            print('estimator.predict failed:', e)
            # try fallback to booster
            try:
                import xgboost as xgb
                if hasattr(estimator, 'get_booster'):
                    booster = estimator.get_booster()
                    dmat = xgb.DMatrix(X_trans)
                    y_pred = booster.predict(dmat)
                    print('predict succeeded via booster')
                else:
                    raise
            except Exception as e2:
                print('fallback booster predict failed:', e2)
                raise

        y_pred = np.asarray(y_pred).ravel()
        rev = TargetValueMapping().reverse_mapping()
        mapped = [rev.get(int(v), str(v)) if (isinstance(v, (int, np.integer)) or isinstance(v, (float, np.floating)) and v.is_integer()) else str(v) for v in y_pred]

        # print sample and accuracy
        n = min(10, len(mapped))
        for i in range(n):
            true = df['class'].astype(str).iloc[i] if 'class' in df.columns else '<no-true>'
            print(f"{i}: pred={mapped[i]} true={true}")

        if 'class' in df.columns:
            true_all = df['class'].astype(str).tolist()
            pred_norm = [str(x).strip() for x in mapped]
            true_norm = [str(x).strip() for x in true_all]
            correct = sum(1 for a,b in zip(pred_norm, true_norm) if a==b)
            acc = correct/len(true_norm)
            print(f"Accuracy: {acc:.4f} ({correct}/{len(true_norm)})")
        else:
            print('No true labels to compute accuracy')

        return 0
    except Exception as e:
        print('ERROR:', e)
        return 3

if __name__ == '__main__':
    sys.exit(main())
