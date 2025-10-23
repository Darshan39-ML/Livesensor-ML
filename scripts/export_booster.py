"""Export XGBoost Booster (JSON + binary) from a saved sklearn wrapper or Booster.

Usage:
  python scripts/export_booster.py --input path/to/wrapper.pkl [--out saved_models/<timestamp>]

If --input is omitted, the script will try to find the latest model under saved_models/<timestamp>/model.pkl
and export to the same timestamp folder.
"""
import argparse
import joblib
import os
import time
from pathlib import Path

import xgboost as xgb


def export_booster_from_wrapper(wrapper_path: Path, out_dir: Path) -> None:
    print(f"Loading wrapper from {wrapper_path}")
    model = joblib.load(wrapper_path)

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        raise RuntimeError("Provided model is not an XGBoost wrapper or Booster")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "model.json"
    bin_path = out_dir / "model.bin"

    print(f"Saving Booster JSON to {json_path}")
    booster.save_model(str(json_path))
    print(f"Saving Booster binary to {bin_path}")
    booster.save_model(str(bin_path))

    print("Export complete")


def find_latest_saved_wrapper(saved_models_dir: Path) -> Path:
    # saved_models/<timestamp>/model.pkl
    candidates = []
    for child in Path(saved_models_dir).iterdir():
        if child.is_dir():
            p = child / "model.pkl"
            if p.exists():
                candidates.append((int(child.name), p))
    if not candidates:
        raise RuntimeError(f"No model.pkl found under {saved_models_dir}")
    latest = max(candidates, key=lambda x: x[0])
    return latest[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to saved wrapper (joblib/pkl)")
    parser.add_argument("--out", help="Output directory; default: saved_models/<timestamp>")
    parser.add_argument("--saved_models_dir", default="saved_models", help="Saved models root dir")
    args = parser.parse_args()

    saved_models_dir = Path(args.saved_models_dir)

    if args.input:
        wrapper_path = Path(args.input)
        if not wrapper_path.exists():
            raise SystemExit(f"Input wrapper not found: {wrapper_path}")
        # set out_dir to sibling timestamp folder if not provided
        timestamp = int(time.time())
        out_dir = Path(args.out) if args.out else saved_models_dir / str(timestamp)
    else:
        wrapper_path = find_latest_saved_wrapper(saved_models_dir)
        # export to same timestamp folder
        out_dir = wrapper_path.parent

    export_booster_from_wrapper(wrapper_path, out_dir)


if __name__ == "__main__":
    main()
