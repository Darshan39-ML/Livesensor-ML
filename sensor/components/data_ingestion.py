"""Data ingestion component: export from data access to CSV feature store and split into train/test."""

from sklearn.model_selection import train_test_split
from pandas import DataFrame
import os
import sys

from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.config_entity import DataIngestionConfig
from sensor.data_access.sensor_data import SensorData


class DataIngestion:
    """Handles extracting data from the data source and producing train/test CSVs.

    Methods raise SensorException on unexpected failures so callers get traceable context.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB to feature store")
            sensor_data = SensorData(collection_name=self.data_ingestion_config.collection_name)

            data_frame = sensor_data.get_collection_as_dataframe()

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            data_frame.to_csv(feature_store_file_path, index=False, header=True)
            return data_frame
        except Exception as e:
            raise SensorException(e, sys) from e

    def split_data_as_train_test(self, data_frame: DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                data_frame, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed train/test split on the data")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting training and testing file paths")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(data_frame=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys) from e

