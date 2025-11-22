# mlops/evaluate/evaluate.py

import json
from pathlib import Path
from ultralytics import YOLO

def evaluate_yolov8(model_path: str, data_yaml: str, output_dir: str = "mlops/evaluate/reports"):
    """
    Evaluate a YOLOv8 model and save metrics to a JSON report.

    Args:
        model_path (str): Path to the trained YOLOv8 .pt model
        data_yaml (str): Path to dataset YAML (train/val/test + classes)
        output_dir (str): Directory to save evaluation report
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO(model_path)

    # Run validation on dataset
    results = model.val(data=data_yaml)

    # Extract key metrics
    metrics = {
        "mAP_0.5": results.metrics.get("map50", None),
        "mAP_0.5_0.95": results.metrics.get("map50_95", None),
        "precision": results.metrics.get("precision", None),
        "recall": results.metrics.get("recall", None),
        "F1": results.metrics.get("f1", None),
        "per_class_AP": results.metrics.get("AP_class", None)
    }

    # Save metrics to JSON
    report_path = output_dir / f"{Path(model_path).stem}_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation complete. Metrics saved to {report_path}")
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 model (.pt)")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--output", default="mlops/evaluate/reports", help="Directory to save JSON report")

    args = parser.parse_args()
    evaluate_yolov8(args.model, args.data, args.output)
