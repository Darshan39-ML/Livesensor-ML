
import os
import sys
import shutil
import numbers

import pandas as pd
from scipy.stats import ks_2samp

from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    """Perform data validation (schema and drift) and produce a DataValidationArtifact.

    This implementation is intentionally conservative: it copies the ingestion
    train/test CSVs into the configured valid/invalid locations depending on
    whether required schema columns are present, and it produces a basic
    drift report using the Kolmogorov-Smirnov test for numeric columns.
    """

    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        try:
            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys) from e
    
    def data_ignestion_config(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        try:
            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise SensorException(e, sys) from e
        
    def data_validation_config(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        try:
            self.config = config
            self.ingestion_artifact = ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise SensorException(e, sys) from e

    def _read_schema(self) -> dict:
        try:
            schema = read_yaml_file(SCHEMA_FILE_PATH)
            if not isinstance(schema, dict):
                return {}
            return schema
        except Exception as e:
            raise SensorException(e, sys) from e
        
    def validate_number_of_columns(self,datafram:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"required number of columns: {number_of_columns}")
            logging.info(f"actual number of columns: {len(datafram.columns)}")
            if number_of_columns == len(datafram.columns):
                return True
            return False
        except Exception as e:
            raise SensorException(e, sys) from e

    def is_numeric_column_exist(self, df: pd.DataFrame, column_name: str) -> bool:
        """Check if a column exists in the DataFrame and is of numeric type."""
        if column_name in df.columns:
            return pd.api.types.is_numeric_dtype(df[column_name])
        return False

    def validate_schema(self) -> bool:
        """Validate that required columns (if present in schema) exist in train/test CSVs.

        Copies files into valid/invalid target paths based on validation result.
        Returns True if both train and test pass schema validation.
        """
        try:
            schema = self._read_schema()
            raw_columns = schema.get("columns") if isinstance(schema, dict) else None

            # normalize schema columns to a set of column names
            expected_set = set()
            if raw_columns:
                if isinstance(raw_columns, dict):
                    expected_set = set(raw_columns.keys())
                elif isinstance(raw_columns, list):
                    for item in raw_columns:
                        if isinstance(item, str):
                            expected_set.add(item)
                        elif isinstance(item, dict):
                            expected_set.update(item.keys())
                        else:
                            try:
                                expected_set.add(str(item))
                            except Exception:
                                continue
                else:
                    try:
                        expected_set = set(raw_columns)
                    except Exception:
                        expected_set = set()

            # Load dataframes from ingestion artifact
            train_path = self.ingestion_artifact.training_file_path
            test_path = self.ingestion_artifact.testing_file_path

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise SensorException(FileNotFoundError("Train or test file missing from ingestion artifact"), sys)

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # If schema doesn't declare columns, treat as pass
            if not expected_set:
                os.makedirs(os.path.dirname(self.config.valid_train_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.config.valid_test_file_path), exist_ok=True)
                shutil.copy(train_path, self.config.valid_train_file_path)
                shutil.copy(test_path, self.config.valid_test_file_path)
                return True

            # If schema didn't provide any columns, treat as pass above
            train_ok = expected_set.issubset(set(train_df.columns))
            test_ok = expected_set.issubset(set(test_df.columns))

            # Ensure directories exist
            os.makedirs(os.path.dirname(self.config.valid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.invalid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.valid_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.invalid_test_file_path), exist_ok=True)

            # Copy to appropriate locations
            if train_ok:
                shutil.copy(train_path, self.config.valid_train_file_path)
            else:
                shutil.copy(train_path, self.config.invalid_train_file_path)

            if test_ok:
                shutil.copy(test_path, self.config.valid_test_file_path)
            else:
                shutil.copy(test_path, self.config.invalid_test_file_path)

            return bool(train_ok and test_ok)
        except Exception as e:
            raise SensorException(e, sys) from e

    def detect_data_drift(self) -> dict:
        """Detect drift between train and test for numeric columns and write a YAML report.

        Returns a dict with per-column p-values and drift boolean.
        """
        try:
            # Prefer validated files if present
            train_path = self.config.valid_train_file_path if os.path.exists(self.config.valid_train_file_path) else self.ingestion_artifact.training_file_path
            test_path = self.config.valid_test_file_path if os.path.exists(self.config.valid_test_file_path) else self.ingestion_artifact.testing_file_path

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            report = {}
            common_cols = set(train_df.columns).intersection(set(test_df.columns))
            for col in sorted(common_cols):
                try:
                    if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(test_df[col]):
                        _, pvalue = ks_2samp(train_df[col].dropna(), test_df[col].dropna())

                        # coerce pvalue to float safely using numbers.Number checks
                        if isinstance(pvalue, numbers.Number):
                            pval = float(pvalue)  # type: ignore
                        elif isinstance(pvalue, (list, tuple)) and len(pvalue) > 0:
                            # try last element
                            last = pvalue[-1]
                            if isinstance(last, numbers.Number):
                                pval = float(last)  # type: ignore
                            else:
                                try:
                                    pval = float(str(last))
                                except (ValueError, TypeError):
                                    continue
                        else:
                            try:
                                pval = float(str(pvalue))
                            except (ValueError, TypeError):
                                continue

                        report[col] = {"p_value": pval, "drift_detected": (pval < 0.05)}
                except (ValueError, TypeError) as e:
                    # skip columns that cause test failures (e.g., non-numeric conversion issues)
                    logging.debug("Skipping column %s due to error: %s", col, e)
                    continue

            # persist the report
            os.makedirs(os.path.dirname(self.config.drift_report_file_path), exist_ok=True)
            write_yaml_file(self.config.drift_report_file_path, report)
            return report
        except Exception as e:
            raise SensorException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            status = self.validate_schema()
            drift_report = self.detect_data_drift()
            logging.debug("Drift report generated with %d entries", len(drift_report) if isinstance(drift_report, dict) else 0)
            artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.config.valid_train_file_path if os.path.exists(self.config.valid_train_file_path) else "",
                valid_test_file_path=self.config.valid_test_file_path if os.path.exists(self.config.valid_test_file_path) else "",
                invalid_train_file_path=self.config.invalid_train_file_path if os.path.exists(self.config.invalid_train_file_path) else "",
                invalid_test_file_path=self.config.invalid_test_file_path if os.path.exists(self.config.invalid_test_file_path) else "",
                drift_report_file_path=self.config.drift_report_file_path,
            )

            train_dataframe = pd.read_csv(artifact.valid_train_file_path)
            test_dataframe = pd.read_csv(artifact.valid_test_file_path)

            status = all(self.is_numeric_column_exist(df=train_dataframe, column_name=col) for col in train_dataframe.columns)
            if not status:
                raise SensorException(Exception("Not all columns in the training data are numeric as required."), sys)

            status = all(self.is_numeric_column_exist(df=test_dataframe, column_name=col) for col in test_dataframe.columns)
            if not status:
                raise SensorException(ValueError("Not all columns in the test data are numeric as required."), sys)

            logging.info("Data validation completed. Artifact: %s", artifact)
            return artifact
        except Exception as e:
            raise SensorException(e, sys) from e

