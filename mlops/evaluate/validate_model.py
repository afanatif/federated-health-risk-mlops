# mlops/evaluate/validate_model.py
import json
import shutil
from pathlib import Path
from typing import Dict

from mlops.evaluate.metrics import compute_metrics

# Repo-root-relative directories
INCOMING_MODELS_DIR = Path("models/incoming")
BLESSED_MODELS_DIR = Path("models/blessed")
REJECTED_MODELS_DIR = Path("models/rejected")
REPORTS_DIR = Path("mlops/evaluate/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical dataset YAML used for evaluation
TEST_DATA_YAML = Path("data/Pothole.v1i.yolov8/data.yaml")

THRESHOLDS = {
    "mAP50": 0.6,
    "mAP50_95": 0.4
}


def validate_model(model_name: str, move_model: bool = True) -> Dict:
    """
    Validate a model located in models/incoming/<model_name> or models/incoming/<model_name>.pt

    Args:
      model_name: folder name under models/incoming OR a .pt filename (e.g. 'yolov8n.pt' or 'yolov8_experiment123')
      move_model: if True, move model into blessed/ or rejected/ after validation

    Returns:
      metrics dict (includes _report_path, _passed, and optionally _moved_to)
    """
    incoming = INCOMING_MODELS_DIR
    model_folder = incoming / model_name

    # Resolve model path: support folder with weights.pt or direct .pt file
    if model_folder.is_dir():
        model_path = model_folder / "weights.pt"
        if not model_path.exists():
            pts = list(model_folder.glob("*.pt"))
            if pts:
                model_path = pts[0]
            else:
                raise FileNotFoundError(f"No .pt model found in {model_folder}")
    else:
        # Maybe a .pt filename under incoming root
        candidate = incoming / model_name
        if candidate.exists() and candidate.suffix == ".pt":
            model_path = candidate
            model_folder = candidate.parent  # for moving later
        else:
            raise FileNotFoundError(f"Model {model_name} not found in {incoming}")

    # Compute metrics and save a report
    metrics = compute_metrics(str(model_path), str(TEST_DATA_YAML), save_report=True, output_dir=REPORTS_DIR, report_name=model_path.stem)

    # Normalize numeric fallback
    mAP50 = metrics.get("mAP50") or 0.0
    mAP50_95 = metrics.get("mAP50_95") or 0.0
    passed = (mAP50 >= THRESHOLDS["mAP50"]) and (mAP50_95 >= THRESHOLDS["mAP50_95"])

    metrics["_passed"] = bool(passed)
    metrics["_report_path"] = str(REPORTS_DIR / f"{model_path.stem}.json") if metrics.get("_report_path") is None else metrics["_report_path"]

    if move_model:
        dest_base = BLESSED_MODELS_DIR if passed else REJECTED_MODELS_DIR
        dest_base.mkdir(parents=True, exist_ok=True)
        print(dest_base)

        # If original was a folder, move folder; else move file
        if (incoming / model_name).is_dir():
            dest = dest_base / (incoming / model_name).name
            shutil.move(str(incoming / model_name), str(dest))
        else:
            dest = dest_base / model_path.name
            shutil.move(str(model_path), str(dest))

        metrics["_moved_to"] = str(dest)

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate model from models/incoming")
    parser.add_argument("model_name", help="model folder name or .pt filename inside models/incoming")
    parser.add_argument("--no-move", dest="move", action="store_false", help="don't move model after validation")
    args = parser.parse_args()
    out = validate_model(args.model_name, move_model=args.move)
    print(json.dumps(out, indent=2))
