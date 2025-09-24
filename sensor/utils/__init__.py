import importlib.util
import os
from typing import Any


# Provide a stub for static analysis / IDEs. At runtime the dynamic loader
# below will (attempt to) replace this with the implementation defined in
# the sibling file `sensor/utils.py`. The stub ensures tools know the symbol
# exists and what its call signature looks like.
def dump_csv_file_to_mongodb_collection(file_path: str, database_name: str, collection_name: str, dry_run: bool = False) -> Any:  # pragma: no cover - runtime replaced
    raise NotImplementedError("This is a runtime-provided implementation")


# Load the top-level helper module `sensor/utils.py` (sibling of this package)
# and re-export the helper functions so code that imports `sensor.utils` (the
# package) can access them at runtime.
_here = os.path.dirname(__file__)
_impl_path = os.path.normpath(os.path.join(_here, '..', 'utils.py'))
if os.path.exists(_impl_path):
    spec = importlib.util.spec_from_file_location("sensor._utils_impl", _impl_path)
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)  # type: ignore
        # re-export the main helper if available
        _impl = getattr(_mod, "dump_csv_file_to_mongodb_collection", None)
        if _impl is not None:
            dump_csv_file_to_mongodb_collection = _impl
