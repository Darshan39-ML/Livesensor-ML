from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from sensor.entity.config_entity import ModelPusherConfig
import os, sys
import shutil   
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.utils.main_utils import save_object, load_object, write_yaml_file


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            model_file_path = self.model_pusher_config.model_file_path

            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            shutil.copy(trained_model_path, model_file_path)

            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(trained_model_path, saved_model_path)

            model_pusher_artifact = ModelPusherArtifact(model_file_path=model_file_path, saved_model_path=saved_model_path)
            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e, sys) from e