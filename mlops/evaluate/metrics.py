# mlops/evaluate/metrics.py

from ultralytics import YOLO
import os

def compute_metrics(model, test_data):
    results = model.val(data=test_data, split="test")

    metrics = {
        "precision": results.box.mp,         # mean precision
        "recall": results.box.mr,            # mean recall
        "mAP50": results.box.map50,          # mAP@0.5
        "mAP50_95": results.box.map,         # mAP@0.5:0.95
        "per_class_map": results.box.maps,   # array of per-class maps
    }

    return metrics

