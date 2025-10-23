#!/usr/bin/env python3
"""
Usage: python3 check_xgb_model.py /path/to/model_file
This script attempts to load common xgboost model formats (native Booster .model/.bin/.json
and sklearn pickles saved with joblib). It prints basic info and tries a quick sanity predict.
"""

import sys
import os
import argparse

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def try_imports():
    try:
        import xgboost as xgb
    except Exception as e:
        xgb = None
    try:
        import joblib
    except Exception as e:
        joblib = None
    import numpy as np
    return xgb, joblib, np

def load_model(path, xgb, joblib):
    # Try native XGBoost Booster load first
    if xgb is not None:
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            return booster
        except Exception:
            pass
    # Try joblib / pickle load
    if joblib is not None:
        try:
            obj = joblib.load(path)
            return obj
        except Exception:
            pass
    # Not loaded
    return None

def inspect_model(model, xgb, np):
    info = {}
    t = type(model).__name__
    info['type'] = t

    # Native Booster
    if xgb is not None and isinstance(model, xgb.core.Booster):
        info['backend'] = 'xgboost.Booster'
        # attempt to get feature names (may be None)
        try:
            fn = model.feature_names
        except Exception:
            fn = None
        info['feature_names'] = fn
        # try number of boosting rounds if available
        n_rounds = None
        try:
            n_rounds = model.num_boosted_rounds()
        except Exception:
            pass
        info['num_boosted_rounds'] = n_rounds
        return info

    # sklearn-style estimator (XGBClassifier or other)
    try:
        params = getattr(model, "get_params", None)
        if callable(params):
            info['backend'] = 'sklearn-like'
            info['params_sample'] = dict(list(model.get_params().items())[:10])
        if hasattr(model, "n_features_in_"):
            info['n_features_in_'] = int(model.n_features_in_)
        if hasattr(model, "feature_names_in_"):
            info['feature_names_in_'] = list(getattr(model, "feature_names_in_"))
        return info
    except Exception:
        return info

def try_predict(model, xgb, np):
    # Return (success, message)
    try:
        # xgboost Booster
        if xgb is not None and isinstance(model, xgb.core.Booster):
            fn = model.feature_names
            if fn is None:
                return False, "Booster has no feature names; cannot create test input."
            n = len(fn)
            data = np.zeros((1, n))
            dmat = xgb.DMatrix(data, feature_names=fn)
            pred = model.predict(dmat)
            return True, f"Predicted output shape: {pred.shape}, sample: {pred[:5].tolist()}"
        # sklearn-like estimator
        if hasattr(model, "predict"):
            if hasattr(model, "n_features_in_"):
                n = int(model.n_features_in_)
                X = np.zeros((1, n))
            else:
                # try to infer from coef_ if present
                if hasattr(model, "coef_"):
                    coef = getattr(model, "coef_")
                    if hasattr(coef, "shape"):
                        n = int(coef.shape[-1])
                        X = np.zeros((1, n))
                    else:
                        return False, "Cannot infer feature count for prediction."
                else:
                    return False, "Model lacks n_features_in_ and coef_; cannot create test input."
            pred = model.predict(X)
            return True, f"Predicted output (sklearn) sample: {str(pred)}"
    except Exception as e:
        return False, f"Prediction failed: {e}"
    return False, "Unsupported model type for prediction."

def main():
    parser = argparse.ArgumentParser(description="Check XGBoost model file and run a quick sanity check.")
    parser.add_argument("model_path", help="Path to model file (xgboost Booster file or joblib pickle)")
    args = parser.parse_args()

    path = args.model_path
    if not os.path.exists(path):
        eprint(f"ERROR: Path does not exist: {path}")
        sys.exit(2)

    xgb, joblib, np = try_imports()
    if xgb is None and joblib is None:
        eprint("ERROR: Neither xgboost nor joblib are importable.")
        eprint("Install with: pip install xgboost joblib")
        sys.exit(3)
    if xgb is None:
        eprint("WARNING: xgboost not importable. Native Booster files won't be loadable.")
        eprint("Install xgboost if you need to load .model/.bin/.json booster files: pip install xgboost")
    if joblib is None:
        eprint("WARNING: joblib not importable. Pickle/joblib saved sklearn models won't be loadable.")
        eprint("Install joblib if you need to load joblib pickles: pip install joblib")

    model = load_model(path, xgb, joblib)
    if model is None:
        eprint("ERROR: Failed to load the model. Supported: xgboost Booster (load_model) or joblib pickles.")
        sys.exit(4)

    info = inspect_model(model, xgb, np)
    print("Model loaded successfully.")
    for k, v in info.items():
        print(f"{k}: {v}")

    ok, msg = try_predict(model, xgb, np)
    if ok:
        print("Quick prediction: OK ->", msg)
        sys.exit(0)
    else:
        print("Quick prediction: SKIPPED/FAILED ->", msg)
        # still success because model was loaded
        sys.exit(0)

if __name__ == "__main__":
    main()
