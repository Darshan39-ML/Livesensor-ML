from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """Artifact produced by the data ingestion component.

    Fields:
        feature_store_file_path: path to the intermediate feature store CSV
        training_file_path: path to the produced training CSV
        testing_file_path: path to the produced testing CSV
    """

    feature_store_file_path: str
    training_file_path: str
    testing_file_path: str
@dataclass
class DataValidationArtifact:
        
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

