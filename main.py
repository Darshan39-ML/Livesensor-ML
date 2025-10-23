import os
import sys
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run

from sensor.logger import logging
from sensor.exception import SensorException
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import load_object
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.constant.application import APP_HOST, APP_PORT
import pandas as pd
from sensor.configuration.mongo_db_connection import MongoDBClient, get_mongodb_connection


app = FastAPI()



origins = ["*"]
#Cross-Origin Resource Sharing (CORS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/",tags=["authentication"])
async def  index():
    return RedirectResponse(url="/docs")





@app.get("/train")
async def train():
    try:
        training_pipeline = TrainPipeline()

        if training_pipeline.is_pipeline_running:
            return Response(content="Training pipeline is already running.", media_type="text/plain")

        training_pipeline.run_pipeline()
        return Response(content="Training successfully completed!", media_type="text/plain")
    except Exception as e:
        logging.exception("Error while running training pipeline")
        return Response(content=f"Error Occurred! {e}", media_type="text/plain", status_code=500)
        




@app.post("/predict")
async def predict(file: Optional[UploadFile] = None):
    """Predict endpoint accepts an uploaded CSV file or uses a default empty dataset placeholder.
    Returns predictions as CSV text.
    """
    try:
        # load input dataframe
        if file is not None:
            contents = await file.read()
            df = pd.read_csv(pd.io.common.BytesIO(contents))
        else:
            return Response(content="No input file provided", media_type="text/plain", status_code=400)

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response(content="Model is not available", media_type="text/plain", status_code=404)

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)

        # Align dataframe columns to the model's expected features (add missing, drop extra, reorder)
        try:
            pre = getattr(model, 'preprocessor', None)
            expected = None
            if pre is not None and hasattr(pre, 'feature_names_in_'):
                expected = list(pre.feature_names_in_)
            else:
                if pre is not None and hasattr(pre, 'named_steps'):
                    for step in pre.named_steps.values():
                        if hasattr(step, 'feature_names_in_'):
                            expected = list(step.feature_names_in_)
                            break
        except Exception:
            expected = None

        if expected is not None:
            missing = [c for c in expected if c not in df.columns]
            extra = [c for c in df.columns if c not in expected]
            if missing:
                for c in missing:
                    df[c] = pd.NA
            if extra:
                df = df.drop(columns=extra)
            df = df[expected]
            # replace pandas NA with numpy nan and coerce numeric columns
            import numpy as _np
            df = df.replace({pd.NA: _np.nan})
            for _c in df.columns:
                try:
                    df[_c] = pd.to_numeric(df[_c], errors='coerce')
                except Exception:
                    pass

        # If model is wrapped (SensorModel) it expects preprocessor inside; otherwise, handle directly
        try:
            y_pred = model.predict(df)
        except Exception:
            # Try to use predict on dataframe values
            y_pred = model.predict(df.values)

        df['predicted_column'] = y_pred
        df['predicted_column'] = df['predicted_column'].replace(TargetValueMapping().reverse_mapping())
        # compute confidence/probability if possible
        confidence = None
        try:
            est = getattr(model, 'model', model)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)
            elif hasattr(est, 'predict_proba'):
                pre = getattr(model, 'preprocessor', None)
                X_for_proba = df
                if pre is not None:
                    X_for_proba = pre.transform(df)
                proba = est.predict_proba(X_for_proba)
            else:
                proba = None

            if proba is not None:
                import numpy as _np
                confidence = _np.max(proba, axis=1)
                df['prediction_confidence_score'] = confidence
                df['prediction_confidence'] = [f"{(s*100):.1f}%" for s in confidence]
        except Exception:
            pass

        csv_bytes = df.to_csv(index=False).encode()

        # prepare overall summary headers: percentage per predicted class and average confidence
        headers = {}
        try:
            # value counts normalized
            counts = df['predicted_column'].value_counts(normalize=True)
            for label, frac in counts.items():
                safe_label = str(label).replace(' ', '_')
                headers[f"X-Predicted-Percent-{safe_label}"] = f"{(frac*100):.1f}%"

            if 'prediction_confidence_score' in df.columns:
                avg_conf = float(df['prediction_confidence_score'].mean())
                headers['X-Average-Confidence-Score'] = f"{avg_conf:.4f}"
                headers['X-Average-Confidence'] = f"{(avg_conf*100):.1f}%"
        except Exception:
            headers = {}

        return Response(content=csv_bytes, media_type='text/csv', headers=headers)
    except SensorException as se:
        logging.exception("SensorException in predict")
        return Response(content=str(se), media_type="text/plain", status_code=500)
    except Exception as e:
        logging.exception("Exception in predict")
        return Response(content=str(e), media_type="text/plain", status_code=500)



@app.get("/health")
async def health():
    """Simple health check that verifies model availability and DB connectivity."""
    status = {"model": "unknown", "mongodb": "unknown"}
    try:
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        status["model"] = "available" if model_resolver.is_model_exists() else "missing"
    except Exception as e:
        status["model"] = f"error: {e}"
    try:
        # try a quick MongoDB ping via the wrapper
        client = MongoDBClient()
        if client.client is None:
            status["mongodb"] = "not_configured"
        elif getattr(client, 'is_in_memory', False):
            status["mongodb"] = "in_memory"
        else:
            # will raise if cannot connect
            client.client.admin.command("ping")
            status["mongodb"] = "ok"
            client.close()
    except Exception as e:
        status["mongodb"] = f"error: {e}"

    return status





def main():
    try:
            
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)



if __name__ == "__main__":

    # file_path="/Users/myhome/Downloads/sensorlive/aps_failure_training_set1.csv"
    # database_name="ineuron"
    # collection_name ="sensor"
    # dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)
    app_run(app, host=APP_HOST, port=APP_PORT)







  












    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)