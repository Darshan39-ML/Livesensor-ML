from dataclasses import dataclass

@dataclass

class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str 




@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


    
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

    
class classification_metric_artifact:
    def __init__(self,f1_score:float,precision_score:float,recall_score:float,accuracy_score:float):
        self.f1_score=f1_score
        self.precision_score=precision_score
        self.recall_score=recall_score
        self.accuracy_score=accuracy_score

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact:classification_metric_artifact
    test_metric_artifact:classification_metric_artifact
    is_model_accepted:bool
    improved_accuracy:float