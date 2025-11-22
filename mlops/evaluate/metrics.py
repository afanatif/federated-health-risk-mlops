# mlops/evaluate/metrics.py
from pathlib import Path
from typing import Union, Dict, Any
import json
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("ultralytics package is required. Install with `pip install ultralytics`.") from e


def _to_python_types(obj):
    """Recursively convert NumPy scalars/arrays to Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python_types(v) for v in obj]
    return obj


def _extract_metrics_from_results(results) -> Dict[str, Any]:
    """
    Normalize metrics extraction across Ultralytics versions.
    Returns dict: precision, recall, mAP50, mAP50_95, per_class_map (list).
    """
    # Preferred: results.metrics (dictionary)
    if hasattr(results, "metrics") and isinstance(results.metrics, dict):
        m = results.metrics
        # keys vary across versions (map50 vs map_50 etc). try common ones.
        def pick(*keys):
            for k in keys:
                if k in m and m[k] is not None:
                    return m[k]
            return None

        precision = pick("precision", "box/precision", "P")
        recall = pick("recall", "box/recall", "R")
        mAP50 = pick("map50", "mAP_0.5", "map_50")
        mAP50_95 = pick("map50_95", "mAP_0.5_0.95", "map_50_95", "map")
        per_class = pick("AP_class", "AP_per_class", "AP", "per_class_AP")

        return {
            "precision": precision,
            "recall": recall,
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "per_class_map": per_class,
        }

    # Fallback: older/newer result object shapes (results.box.*)
    try:
        box = results.box
        return {
            "precision": getattr(box, "mp", None),
            "recall": getattr(box, "mr", None),
            "mAP50": getattr(box, "map50", None),
            "mAP50_95": getattr(box, "map", None),
            "per_class_map": getattr(box, "maps", None),
        }
    except Exception:
        # Last resort: return whatever is present on results
        return _to_python_types(getattr(results, "__dict__", {}))


def compute_metrics(model_or_path: Union[str, YOLO],
                    data_yaml: str,
                    save_report: bool = False,
                    output_dir: Union[str, Path] = "mlops/evaluate/reports",
                    report_name: str = None) -> Dict[str, Any]:
    """
    Compute YOLOv8 metrics.

    Arg1s:
      model_or_path: Either an Ultralytics YOLO object or a filesystem path to a .pt model.
      data_yaml: Path to dataset YAML (must include train/val/test keys).
      save_report: If True, save a JSON report to output_dir.
      output_dir: Directory to save the JSON report.
      report_name: Optional filename (without extension). If omitted uses model stem.

    Returns:
      metrics: plain-Python dict with keys:
        precision, recall, mAP50, mAP50_95, per_class_map
    """
    # Ensure data_yaml is a string path
    data_yaml = str(data_yaml)
    # Load model if a path was given
    if isinstance(model_or_path, str):
        model = YOLO(model_or_path)
    else:
        model = model_or_path

    # Run validation (Ultralytics handles YAML and splits)
    results = model.val(data=data_yaml)

    metrics = _extract_metrics_from_results(results)
    metrics = _to_python_types(metrics)  # make safe for JSON

    if save_report:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if report_name:
            filename = f"{report_name}.json"
        else:
            # try to infer a name from the model path if available
            try:
                model_path = model_or_path if isinstance(model_or_path, str) else getattr(model, "model", None)
                stem = Path(model_path).stem if isinstance(model_path, (str, Path)) else "model_eval"
            except Exception:
                stem = "model_eval"
            filename = f"{stem}_eval_report.json"
        report_path = out_dir / filename
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        metrics["_report_path"] = str(report_path)

    return metrics
