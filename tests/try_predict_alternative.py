import os
import sys
import pandas as pd
import numpy as np

from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object, load_model_flexible
from sensor.constant.training_pipeline import SAVED_MODEL_DIR


def main():
    resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
    model_path = resolver.get_best_model_path()
    print('Model path:', model_path)
    model = load_object(file_path=model_path)
    print('Wrapper class:', type(model))
    estimator = getattr(model, 'model', None)
    pre = getattr(model, 'preprocessor', None)
    print('Estimator type:', type(estimator))
    print('Estimator has get_booster:', hasattr(estimator, 'get_booster'))
    print('Estimator is xgboost Booster instance:', type(estimator).__name__)
    print('Has attribute _xgb_raw_state:', hasattr(estimator, '_xgb_raw_state'))

    # expected features
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
    print('Expected feature count:', len(expected) if expected is not None else None)

    sample_csv = 'artifact/09_30_2025_21_13_38/data_ingestion/feature_store/sensor.csv'
    df = pd.read_csv(sample_csv)
    print('CSV columns:', len(df.columns))

    # build X matching expected features
    if expected is not None:
        # ensure all expected present, add NA for missing
        X = df.copy()
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]
        print('Missing columns (added as NA):', missing[:10])
        print('Extra columns (will be dropped):', extra[:10])
        for c in missing:
            X[c] = pd.NA
        X = X[expected]
    else:
        X = df.drop(columns=['class']) if 'class' in df.columns else df

    # run preprocessor transform
    try:
        X_trans = pre.transform(X)
        print('Transformed shape:', X_trans.shape)
    except Exception as e:
        print('Preprocessor transform failed:', e)
        return 3

    # Try sklearn-like predict
    try:
        y_hat = estimator.predict(X_trans)
        print('Estimator.predict succeeded, sample:', np.asarray(y_hat).ravel()[:5])
        return 0
    except Exception as e:
        print('Estimator.predict failed:', e)

    # Try to get booster and use xgboost DMatrix
    try:
        import xgboost as xgb
    except Exception:
        xgb = None
    if xgb is not None:
        try:
            if hasattr(estimator, 'get_booster'):
                booster = estimator.get_booster()
                dmat = xgb.DMatrix(X_trans)
                pred = booster.predict(dmat)
                print('Booster predict via get_booster() succeeded, sample:', pred[:5])
                return 0
            else:
                # maybe estimator is a Booster already
                if isinstance(estimator, xgb.Booster):
                    dmat = xgb.DMatrix(X_trans)
                    pred = estimator.predict(dmat)
                    print('Estimator is Booster and predict succeeded, sample:', pred[:5])
                    return 0
        except Exception as e:
            print('Booster prediction failed:', e)

    # As a last resort, try to load model file as raw booster
    try:
        raw = load_model_flexible(model_path)
        print('Flexible loader returned type:', type(raw))
        if xgb is not None and isinstance(raw, xgb.Booster):
            dmat = xgb.DMatrix(X_trans)
            pred = raw.predict(dmat)
            print('Flexible-loaded booster predict sample:', pred[:5])
            return 0
    except Exception as e:
        print('Flexible load failed:', e)

    print('All prediction attempts failed')
    return 4

if __name__ == '__main__':
    sys.exit(main())
