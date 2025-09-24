import yaml
import os
import sys
from typing import Any, Optional

try:
    import dill
except ImportError:  # pragma: no cover - dill optional at import time
    dill = None

from sensor.exception import SensorException


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML file and return the parsed content as a dict.

    Returns an empty dict if the file exists but is empty. Raises SensorException on IO or parse errors.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise SensorException(FileNotFoundError(f"YAML file not found: {file_path}"), sys) from None
    except Exception as e:
        raise SensorException(e, sys) from e


def write_yaml_file(file_path: str, data: Any) -> None:
    """Write `data` to `file_path` as YAML.

    Creates parent directories as needed. Raises SensorException on error.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh)
    except Exception as e:
        raise SensorException(e, sys) from e


def save_object(file_path: str, obj: Any) -> None:
    """Serialize Python object to disk using dill (fallback to pickle if dill missing).

    Raises SensorException on failure.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if dill is not None:
            with open(file_path, "wb") as fh:
                dill.dump(obj, fh)
        else:
            # lightweight fallback: use pickle from stdlib
            import pickle

            with open(file_path, "wb") as fh:
                pickle.dump(obj, fh)
    except Exception as e:
        raise SensorException(e, sys) from e


def load_object(file_path: str) -> Optional[Any]:
    """Load a Python object previously saved with save_object.

    Returns None if file does not exist. Raises SensorException on other errors.
    """
    try:
        if not os.path.exists(file_path):
            return None
        if dill is not None:
            with open(file_path, "rb") as fh:
                return dill.load(fh)
        else:
            import pickle

            with open(file_path, "rb") as fh:
                return pickle.load(fh)
    except Exception as e:
        raise SensorException(e, sys) from e