from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
import os
import sys
from sensor.exception import SensorException
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from logging import getLogger
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation

class TrainPipeline:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        self.training_pipeline_config = training_pipeline_config
        self.logger = getLogger(__name__)
        self.logger.info(f"TrainingPipelineConfig: {self.training_pipeline_config}")

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.logger.info("Starting data ingestion...")
            data_ingestion = DataIngestion(self.training_pipeline_config.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            self.logger.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            self.logger.info("Starting data validation...")
            data_validation = DataValidation(
                self.training_pipeline_config.data_validation_config,
                data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            self.logger.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys) from e

    def run_pipeline(self) -> None:
        try:
            self.logger.info("Training pipeline started.")
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact: DataValidationArtifact = self.start_data_validation(data_ingestion_artifact)

            data_ingestion = DataIngestion(
                self.training_pipeline_config.data_ingestion_config,
                data_ingestion_artifact
            )

            data_validation = DataValidation(
                self.training_pipeline_config.data_validation_config,
                data_ingestion_artifact
            )
            data_validation_artifact: DataValidationArtifact = self.start_data_validation(data_ingestion_artifact)
            # Further steps like data transformation, model training, etc. would go here.
            self.logger.info("Training pipeline completed successfully.")
        except Exception as e:
            raise SensorException(e, sys) from e
        
