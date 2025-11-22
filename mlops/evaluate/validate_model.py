# mlops/evaluate/validate_model.py

import os
import shutil
from model_service.core.model_loader import load_model
from evaluate.metrics import compute_metrics

INCOMING_MODELS_DIR = "../../models/incoming"
BLESSED_MODELS_DIR = "../../models/blessed"
REJECTED_MODELS_DIR = "../../models/rejected"
TEST_DATA_DIR = "../../federated_training/datasets/test"
THRESHOLDS = {
    "mAP50": 0.6,
    "mAP50_95": 0.4
}

def validate_model(model_name):
    model_path = os.path.join(INCOMING_MODELS_DIR, model_name, "weights.pt")
    model = load_model(model_path)

    metrics = compute_metrics(model, TEST_DATA_DIR)

    if metrics["mAP50"] >= THRESHOLDS["mAP50"] and metrics["mAP50_95"] >= THRESHOLDS["mAP50_95"]:
        dest = os.path.join(BLESSED_MODELS_DIR, model_name)
        shutil.move(os.path.dirname(model_path), dest)
        print(f"Model {model_name} validated and moved to blessed models.")
    else:
        dest = os.path.join(REJECTED_MODELS_DIR, model_name)
        shutil.move(os.path.dirname(model_path), dest)
        print(f"Model {model_name} failed validation and moved to rejected models.")

    # Optionally: save metrics report
    report_path = os.path.join("reports", f"{model_name}_metrics.json")
    with open(report_path, "w") as f:
        import json
        json.dump(metrics, f, indent=4)
