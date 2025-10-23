import os
import sys
import pandas as pd
import numpy as np

from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR


def main():
    resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
    model_path = resolver.get_best_model_path()
    model = load_object(file_path=model_path)
    print('Model class:', type(model))
    pre = getattr(model, 'preprocessor', None)
    estimator = getattr(model, 'model', None)
    print('Has preprocessor:', pre is not None)
    if pre is not None:
        # try to find expected feature names
        expected = None
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
        print('Expected features count:', len(expected) if expected is not None else 'unknown')

    sample_csv = 'artifact/09_30_2025_21_13_38/data_ingestion/feature_store/sensor.csv'
    df = pd.read_csv(sample_csv)
    print('Sample CSV columns count:', df.shape[1])
    print('Sample CSV first columns:', df.columns[:10].tolist())

    if pre is not None and expected is not None:
        print('\nDifference:')
        csv_cols = list(df.columns)
        missing = [c for c in expected if c not in csv_cols]
        extra = [c for c in csv_cols if c not in expected]
        print('Missing from CSV (in model):', missing[:20])
        print('Extra in CSV (not in model):', extra[:20])

if __name__ == '__main__':
    main()
