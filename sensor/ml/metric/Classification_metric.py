from sensor.entity.artifact_entity import classification_metric_artifact
from sensor.exception import SensorException
import sys,os
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

def get_classification_score(y_true,y_pred)->classification_metric_artifact:
    try:
        f1=f1_score(y_true,y_pred)
        precision=precision_score(y_true,y_pred)
        recall=recall_score(y_true,y_pred)
        accuracy=accuracy_score(y_true,y_pred)

        classification_metric=classification_metric_artifact(f1,precision,recall,accuracy)

        return classification_metric

    except Exception as e:
        raise SensorException(e,sys) from e