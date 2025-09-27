from sensor.utils.main_utils import load_numpy_array_data
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
from xgboost import XGBClassifier
from sensor.ml.metric.Classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys) from e 
        
    def perform_hyper_parameter_tuning(self):...
       
    def train_model(self,x,y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e,sys) from e
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            train_array=load_numpy_array_data(train_file_path)
            test_array=load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            model = self.train_model(x_train,y_train)
            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            
            if classification_train_metric.f1_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give expected accuracy: {classification_train_metric.f1_score} < {self.model_trainer_config.expected_score}")
            
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception(f"Model is not good as it is not able to give expected accuracy: {diff} > {self.model_trainer_config.overfitting_underfitting_threshold}")
            
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir,exist_ok=True)
            sensor_model = SensorModel(model=model)
            sensor_model.save_model(self.model_trainer_config.trained_model_file_path)
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys) from e