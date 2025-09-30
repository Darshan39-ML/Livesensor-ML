from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os , sys
from sensor.logger import logging
from sensor.constant.application import APP_HOST,APP_PORT
from fastapi import FastAPI
from sensor.entity.config_entity  import TrainingPipelineConfig,DataIngestionConfig
import warnings
# Filter out the specific FutureWarning from xgboost
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
from sensor.pipeline.training_pipeline import TrainPipeline
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.utils.main_utils import load_object
from fastapi import File, UploadFile, responses
from fastapi.responses import Response
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping


# def test_exception():
#     try:
#         logging.info("ki yaha p bhaiaa ek error ayegi diveision by zero wali error ")
#         a=1/0
#     except Exception as e:
#        raise SensorException(e,sys) 

app = FastAPI()


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train():
    try:
        training_pipeline = TrainPipeline()
        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running")
        
        training_pipeline.run_pipeline()
        return Response("training successfuly completed")

    except Exception as e:
        return Response(f"error occured :"{e})
    



@app.get("/predict")




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
    app_run(app,host=APP_HOST,port=APP_PORT)









  












    # try:
    #     test_exception()
    # except Exception as e:
    #     print(e)