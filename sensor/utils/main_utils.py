import yaml
import pandas as pd
import numpy as np
import os
import pickle
import sys
from sensor.exception import SensorException
from sensor.logger import logging


def read_yaml_file(file_path:str)->dict:
    try:

        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        

    except Exception as e:
        raise SensorException(e,sys)
    


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise SensorException(e, sys)



def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise SensorException(e, sys) from e
    


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise SensorException(e, sys) from e
    



def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise SensorException(e, sys) from e
    



def load_object(file_path: str, ) -> object:
    # Try to load a pickled object first; if that fails, attempt flexible
    # model loading (xgboost native Booster or joblib/pickle wrappers).
    try:
        import pkg_resources
    except ImportError as e:
        raise RuntimeError(
            "pkg_resources (from setuptools) is required but not installed. "
            "Install it in your environment: pip install setuptools"
        ) from e

    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")

        # First attempt: unpickle (common for sklearn-wrapped models)
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)
        except Exception as pickle_err:
            # If pickle triggered an XGBoost-specific error (happens when the
            # pickled object contains xgboost.Booster state that cannot be
            # restored e.g. due to version mismatch or corrupted content),
            # detect that and raise a clearer message for the UI.
            pickle_err_repr = repr(pickle_err)
            err_type_name = type(pickle_err).__name__
            is_xgb_err = (
                "xgboost" in pickle_err_repr.lower()
                or "xgboosterror" in err_type_name.lower()
                or "XGBoost" in pickle_err_repr
            )
            if is_xgb_err:
                # Try a safe unpickle where xgboost.Booster.__setstate__ is
                # monkeypatched to avoid calling into the C library. This can
                # recover the Python wrapper object (e.g., preprocessor) even
                # if the Booster's binary state is incompatible.
                try:
                    def _safe_unpickle(path):
                        try:
                            import xgboost as _xgb
                            core = getattr(_xgb, 'core', None)
                            if core is None or not hasattr(core, 'Booster'):
                                # nothing to patch
                                with open(path, 'rb') as f:
                                    return pickle.load(f)

                            Booster = core.Booster
                            original_setstate = getattr(Booster, '__setstate__', None)

                            def _no_parse_setstate(self, state):
                                # store raw state and don't parse into libxgboost
                                try:
                                    object.__setattr__(self, '_xgb_raw_state', state)
                                except Exception:
                                    pass

                            # patch
                            Booster.__setstate__ = _no_parse_setstate
                            try:
                                with open(path, 'rb') as f:
                                    obj = pickle.load(f)
                            finally:
                                # restore
                                if original_setstate is not None:
                                    Booster.__setstate__ = original_setstate
                            return obj
                        except Exception:
                            # If anything fails, re-raise original pickle_err
                            raise

                    recovered = _safe_unpickle(file_path)
                    # Return the partially recovered object. Note: the Booster
                    # inside may have attribute '_xgb_raw_state' instead of a
                    # working model; callers should handle that.
                    return recovered
                except Exception:
                    raise SensorException(
                        Exception(
                            "Model file appears to contain XGBoost state that cannot be loaded. "
                            "This may be due to an incompatible xgboost version or corrupted file. "
                            "Try re-saving the model with the running xgboost version or use a matching environment."
                        ),
                        sys,
                    ) from pickle_err

            # Otherwise try to load via the flexible loader (joblib or native booster)
            try:
                return load_model_flexible(file_path)
            except Exception as flex_err:
                # Raise a SensorException that includes the original pickle
                # and flexible loader messages for debugging.
                raise SensorException(
                    Exception(f"Failed to load object. pickle error: {pickle_err}; flexible loader error: {flex_err}"),
                    sys,
                ) from flex_err

    except Exception as e:
        raise SensorException(e, sys) from e


# Replace deprecated pkg_resources import with a compatibility shim

try:
    # prefer stdlib importlib.metadata (Python 3.8+)
    from importlib.metadata import version as _dist_version, entry_points as _entry_points
    def get_distribution_version(pkg_name):
        try:
            return _dist_version(pkg_name)
        except Exception:
            return None
    def get_entry_points(group=None):
        try:
            eps = _entry_points()
            if group is None:
                return eps
            # importlib.metadata.entry_points() may return different shapes across versions
            try:
                return eps.select(group=group)
            except Exception:
                # older API: eps is a dict-like
                return eps.get(group, [])
        except Exception:
            return []
except Exception:
    try:
        # try the importlib_metadata backport
        from importlib_metadata import version as _dist_version, entry_points as _entry_points  # type: ignore
        def get_distribution_version(pkg_name):
            try:
                return _dist_version(pkg_name)
            except Exception:
                return None
        def get_entry_points(group=None):
            try:
                eps = _entry_points()
                if group is None:
                    return eps
                try:
                    return eps.select(group=group)
                except Exception:
                    return eps.get(group, [])
            except Exception:
                return []
    except Exception:
        # last resort: use pkg_resources if available
        try:
            import pkg_resources  # type: ignore
            def get_distribution_version(pkg_name):
                try:
                    return pkg_resources.get_distribution(pkg_name).version
                except Exception:
                    return None
            def get_entry_points(group=None):
                if group is None:
                    return []
                try:
                    return list(pkg_resources.iter_entry_points(group))
                except Exception:
                    return []
        except Exception:
            # very last fallback: no metadata available
            def get_distribution_version(pkg_name):
                return None
            def get_entry_points(group=None):
                return []


# ...existing code...


def load_model_flexible(path):
    """
    Try loading an XGBoost model saved in either native xgboost format (json/bin)
    or a pickle/joblib object (sklearn wrapper). Returns the loaded object.
    """
    # Decide strategy based on file extension and a small file magic sniff.
    ext = os.path.splitext(path)[1].lower()

    # Helper: try joblib/pickle load
    def _try_joblib():
        try:
            import joblib
            return joblib.load(path)
        except Exception:
            return None

    # If file extension clearly indicates a booster or JSON, try xgboost first.
    try:
        import xgboost as xgb
    except Exception:
        xgb = None

    # Quick magic sniff (read a few bytes)
    magic = b""
    try:
        with open(path, "rb") as f:
            magic = f.read(64)
    except Exception:
        magic = b""

    # If magic looks like a pickle (protocol 4/5 start 0x80) or extension suggests pickle, prefer joblib
    if ext in (".pkl", ".joblib", ".pickle") or (len(magic) > 0 and magic[0] == 0x80):
        obj = _try_joblib()
        if obj is not None:
            return obj

    # If extension suggests xgboost booster or JSON, try Booster
    if xgb is not None and ext in (".model", ".bin", ".json") or (magic.strip().startswith(b"{") and xgb is not None):
        try:
            booster = xgb.Booster()
            # xgb.Booster.load_model can raise low-level C++ errors; catch them and wrap
            booster.load_model(path)
            return booster
        except Exception as e:
            # If booster load fails, fall through to joblib attempt below
            booster_err = e
    else:
        booster_err = None

    # Last resort: try joblib/pickle load
    obj = _try_joblib()
    if obj is not None:
        return obj

    # No loader succeeded: raise clear error including any booster error
    if booster_err is not None:
        raise RuntimeError(f"Failed to load model as xgboost Booster: {booster_err}")
    raise RuntimeError(f"Failed to load model from {path}: unsupported format or corrupted file")
# ...existing code...


def _check_xgb_model_cli():
    """Command-line helper to inspect an xgboost / joblib model file.

    This used to live at module top-level and executed during import which
    broke tests and library imports. Keep it here under a guarded entrypoint
    so importing this module is side-effect free.
    """
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Check XGBoost model file and run a quick sanity check.")
    parser.add_argument("model_path", help="Path to model file (xgboost Booster file or joblib pickle)")
    args = parser.parse_args()

    path = args.model_path
    if not os.path.exists(path):
        print(f"ERROR: Path does not exist: {path}", file=sys.stderr)
        raise SystemExit(2)

    # try to import optional deps and inspect/load
    try:
        import xgboost as xgb
    except Exception:
        xgb = None
    try:
        import joblib
    except Exception:
        joblib = None
    import numpy as np

    if xgb is None and joblib is None:
        print("ERROR: Neither xgboost nor joblib are importable.", file=sys.stderr)
        print("Install with: pip install xgboost joblib", file=sys.stderr)
        raise SystemExit(3)

    if xgb is None:
        print("WARNING: xgboost not importable. Native Booster files won't be loadable.", file=sys.stderr)
    if joblib is None:
        print("WARNING: joblib not importable. Pickle/joblib saved sklearn models won't be loadable.", file=sys.stderr)

    # attempt to load
    model = None
    if xgb is not None:
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            model = booster
        except Exception:
            model = None
    if model is None and joblib is not None:
        try:
            model = joblib.load(path)
        except Exception:
            model = None

    if model is None:
        print("ERROR: Failed to load the model. Supported: xgboost Booster (load_model) or joblib pickles.", file=sys.stderr)
        raise SystemExit(4)

    print("Model loaded successfully.")
    # minimal inspection
    t = type(model).__name__
    print("type:", t)

    # try a quick predict where reasonable
    try:
        if xgb is not None and isinstance(model, xgb.core.Booster):
            fn = getattr(model, "feature_names", None)
            if fn is None:
                print("Booster has no feature names; cannot create test input.")
            else:
                n = len(fn)
                data = np.zeros((1, n))
                dmat = xgb.DMatrix(data, feature_names=fn)
                pred = model.predict(dmat)
                print("Quick prediction OK, output shape:", getattr(pred, 'shape', None))
        elif hasattr(model, "predict"):
            if hasattr(model, "n_features_in_"):
                n = int(model.n_features_in_)
                X = np.zeros((1, n))
            elif hasattr(model, "coef_"):
                coef = getattr(model, "coef_")
                try:
                    n = int(coef.shape[-1])
                    X = np.zeros((1, n))
                except Exception:
                    X = None
            else:
                X = None
            if X is not None:
                pred = model.predict(X)
                print("Quick prediction (sklearn-like) OK, sample:", str(pred))
            else:
                print("Cannot infer input shape for sklearn-like model; skipping quick predict.")
    except Exception as e:
        print("Prediction failed:", e)


if __name__ == "__main__":
    _check_xgb_model_cli()


