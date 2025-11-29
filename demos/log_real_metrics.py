"""Log real YOLOv8 metrics to MLflow"""
import mlflow

# Your actual metrics from training
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("yolov8-health-risk-detection")

with mlflow.start_run(run_name="real_training_metrics"):
    # Log parameters
    mlflow.log_params({
        "model": "YOLOv8",
        "layers": 72,
        "parameters": 3007013,
        "gflops": 8.1,
        "device": "CUDA:0 RTX 3050",
        "images": 285,
        "instances": 659
    })
    
    # Log your actual validation metrics
    mlflow.log_metrics({
        "mAP50": 0.8269,
        "mAP50-95": 0.5548,
        "precision": 0.8159,
        "recall": 0.7691,
        "box_loss": 0.0  # Add if you have it
    })
    
    # Per-class metrics
    classes = {
        "banner": {"precision": 0.744, "recall": 0.956, "mAP50": 0.938, "mAP50-95": 0.665},
        "erosion": {"precision": 0.732, "recall": 0.944, "mAP50": 0.960, "mAP50-95": 0.689},
        "hcrack": {"precision": 0.726, "recall": 0.418, "mAP50": 0.614, "mAP50-95": 0.327},
        "pothole": {"precision": 1.0, "recall": 0.448, "mAP50": 0.554, "mAP50-95": 0.328},
        "trash": {"precision": 0.844, "recall": 1.0, "mAP50": 0.995, "mAP50-95": 0.681},
        "vcrack": {"precision": 0.850, "recall": 0.848, "mAP50": 0.900, "mAP50-95": 0.638}
    }
    
    for class_name, metrics in classes.items():
        mlflow.log_metrics({
            f"{class_name}_precision": metrics["precision"],
            f"{class_name}_recall": metrics["recall"],
            f"{class_name}_mAP50": metrics["mAP50"],
            f"{class_name}_mAP50-95": metrics["mAP50-95"]
        })
    
    print("✅ Real metrics logged to MLflow!")
    print("🌐 View at: http://localhost:5000")
