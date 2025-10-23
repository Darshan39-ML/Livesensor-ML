import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sensor.ml.model.estimator import SensorModel
from sensor.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME


def main():
    sample_csv = 'artifact/09_30_2025_21_13_38/data_ingestion/feature_store/sensor.csv'
    df = pd.read_csv(sample_csv)

    if 'class' not in df.columns:
        raise SystemExit('No target column class in sample CSV')

    y = df['class'].astype(str).map({'pos':1,'neg':0}).fillna(0).astype(int)
    X = df.drop(columns=['class'])

    # Use only columns that are numeric; for simplicity, coerce non-numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Simple pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    clf = LogisticRegression(max_iter=1000)

    preprocessor.fit(X)
    Xt = preprocessor.transform(X)
    clf.fit(Xt, y)

    # wrap like SensorModel
    model_wrapper = SensorModel(preprocessor=preprocessor, model=clf)

    # save in new timestamp dir
    ts = str(int(time.time()))
    dir_path = os.path.join(SAVED_MODEL_DIR, ts)
    os.makedirs(dir_path, exist_ok=True)
    model_path = os.path.join(dir_path, MODEL_FILE_NAME)
    with open(model_path, 'wb') as f:
        pickle.dump(model_wrapper, f)

    print('Saved new model to', model_path)

if __name__ == '__main__':
    main()
