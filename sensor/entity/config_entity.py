# ...existing code...
from datetime import datetime
from typing import Optional
import os
from sensor.constant import training_pipeline

# helper to ensure required constants exist in training_pipeline
_REQUIRED_ATTRS = [
    "PIPELINE_NAME",
    "ARTIFACT_DIR",
    "DATA_INGESTION_DIR_NAME",
    "DATA_INGESTION_FEATURE_STORE_DIR",
    "FILE_NAME",
    "DATA_INGESTION_INGESTED_DIR",
    "TRAIN_FILE_NAME",
    "TEST_FILE_NAME",
    "DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO",
    "DATA_INGESTION_COLLECTION_NAME",
]

_missing = [a for a in _REQUIRED_ATTRS if not hasattr(training_pipeline, a)]
if _missing:
    raise AttributeError(
        f"Missing required attributes in sensor.constant.training_pipeline: {', '.join(_missing)}"
    )


class TrainingPipelineConfig:
    def __init__(self, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp_str)
        self.timestamp: str = timestamp_str


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )

        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )

        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
        )

        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
# ...existing code...


    


    
   
   



