# from sensor.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os
class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    


class SensorModel:
    def __init__(self, preprocessing, model):
        try:
            self.preprocessing = preprocessing
            self.model = model
        except Exception as e:
            raise e
    
    def predict(self, X):
        try:
            X_transformed = self.preprocessing.transform(X)
            y_hat = self.model.predict(X_transformed)
            return y_hat
        except Exception as e:
            raise e