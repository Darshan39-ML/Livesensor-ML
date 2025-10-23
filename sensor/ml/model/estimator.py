from sensor.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME


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

    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise e
    
    def predict(self,x):
        try:
            # If input is a pandas DataFrame, try to align its columns to what
            # the preprocessor was fitted with. This helps when uploaded CSVs
            # contain extra/renamed columns or when feature ordering differs.
            import pandas as _pd

            if hasattr(x, 'columns'):
                X = x.copy()

                expected = None
                # common place where sklearn stores feature names
                if hasattr(self.preprocessor, 'feature_names_in_'):
                    expected = list(self.preprocessor.feature_names_in_)
                else:
                    # pipeline-like preprocessor: look for a step with feature_names_in_
                    try:
                        if hasattr(self.preprocessor, 'named_steps'):
                            for step in self.preprocessor.named_steps.values():
                                if hasattr(step, 'feature_names_in_'):
                                    expected = list(step.feature_names_in_)
                                    break
                    except (AttributeError, TypeError):
                        expected = None

                if expected is not None:
                    # add missing columns with NA and drop extras, then reorder
                    missing = [c for c in expected if c not in X.columns]
                    extra = [c for c in X.columns if c not in expected]
                    if missing:
                        for c in missing:
                            X[c] = _pd.NA
                    if extra:
                        X = X.drop(columns=extra)
                    # ensure column order matches expected
                    X = X[expected]

                x_transform = self.preprocessor.transform(X)
            else:
                # numpy array or other array-like
                x_transform = self.preprocessor.transform(x)

            # Compatibility shim for XGBoost sklearn wrappers saved with older
            # versions: newer xgboost may have removed attributes like
            # `use_label_encoder` which some pickled estimators expect to
            # exist when get_params() is called. Add a safe default if
            # missing to avoid AttributeError during prediction.
            try:
                cls_name = getattr(self.model, '__class__').__name__
                if cls_name.startswith('XGB'):
                    # Best-effort compatibility shims for attributes that may be
                    # missing between different xgboost versions.
                    for attr, val in (
                        ('use_label_encoder', False),
                        ('gpu_id', None),
                        ('predictor', None),
                        ('n_jobs', 1),
                    ):
                        if not hasattr(self.model, attr):
                            try:
                                setattr(self.model, attr, val)
                            except (AttributeError, TypeError):
                                pass
            except (AttributeError, TypeError):
                # ignore any issues inspecting the model
                pass

            try:
                y_hat = self.model.predict(x_transform)
                return y_hat
            except AttributeError as ae:
                # Some pickled XGBoost sklearn wrappers (older/newer versions)
                # may be missing attributes expected by get_params() which
                # causes AttributeError. Try to fall back to the raw Booster
                # prediction using DMatrix when possible.
                try:
                    import xgboost as _xgb
                    # get_booster exists on sklearn wrappers
                    if hasattr(self.model, 'get_booster'):
                        booster = self.model.get_booster()
                    elif isinstance(self.model, _xgb.Booster):
                        booster = self.model
                    else:
                        raise

                    dm = _xgb.DMatrix(x_transform)
                    y_hat = booster.predict(dm)
                    return y_hat
                except Exception:
                    # Re-raise the original attribute error wrapped in our
                    # Prediction failed message for clarity.
                    raise type(ae)(f"Prediction failed (sklearn wrapper error): {ae}") from ae
        except Exception as e:
            # If this looks like an XGBoost Booster incompatibility (e.g.
            # missing internal attributes like `handle`), raise a clearer,
            # actionable error so the UI can display guidance instead of a
            # raw stack trace.
            msg = str(e)
            if 'handle' in msg or 'Invalid cast' in msg or 'XGBoostError' in msg:
                raise RuntimeError(
                    "Model prediction failed due to XGBoost artifact incompatibility. "
                    "This commonly happens when a Booster was pickled with a different "
                    "xgboost/libxgboost version than the one installed.\n"
                    "Workarounds: re-export the model with the current xgboost version, "
                    "or provide the model as a raw Booster file (JSON/BIN) saved with the "
                    "matching xgboost runtime."
                ) from e
            # Re-raise with a clearer message while preserving original traceback
            raise type(e)(f"Prediction failed: {e}") from e
        

class ModelResolver: 

    
    def __init__(self,model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise e    


    def get_best_model_path(self,)->str:
        try:
            timestamps = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            
            latest_model_path= os.path.join(self.model_dir,f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise e   
        


    def is_model_exists(self)->bool:
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps)==0:
                return False
            
            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise e
