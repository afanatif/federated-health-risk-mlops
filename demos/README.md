# Demo Scripts

This directory contains demonstration scripts for testing MLflow and metrics logging without running full federated training.

## Scripts

### `federated_3_rounds.py` ⭐ **RECOMMENDED**
Simulates 3 federated learning rounds with realistic metric progression. Shows improvement over rounds with line graphs in MLflow.

**Usage:**
```bash
docker cp demos/federated_3_rounds.py fl-mlflow:/tmp/
docker exec fl-mlflow python /tmp/federated_3_rounds.py
```

**What it logs:**
- mAP50, mAP50-95, Precision, Recall
- Train/Val Loss progression
- Per-class metrics for all 6 classes
- 3 rounds showing improvement

### `log_real_metrics.py`
Logs your actual YOLOv8 validation metrics to MLflow (single snapshot).

### `federated_training.py`
Full production-ready federated learning simulation (269 lines) with realistic metrics.

### `simple_demo.py`
Simplified version for quick testing.

### `local_demo.py`
Minimal demo that runs locally (29 lines).

### `demo_mlops.py`
Original demo script with Prometheus integration.

## Quick Start

```powershell
# Ensure MLflow is running
docker-compose up -d mlflow

# Run the 3-round federated demo
docker cp demos/federated_3_rounds.py fl-mlflow:/tmp/
docker exec fl-mlflow python /tmp/federated_3_rounds.py

# Open MLflow UI
Start-Process "http://localhost:5000"
```

## View Results

1. Navigate to http://localhost:5000
2. Click on experiment `yolov8-health-risk-detection`
3. Click on run to see line graphs
4. Compare multiple runs side-by-side
