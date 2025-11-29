# MLOps Demo - Quick Start Guide

## Overview
This demo simulates federated learning with fake metrics to showcase the complete MLOps pipeline:
- ✅ MLflow experiment tracking
- ✅ Prometheus metrics
- ✅ Federated aggregation simulation
- ✅ No real training data needed!

## Prerequisites

Make sure MLflow is running:
```powershell
docker-compose up -d mlflow
```

## Run the Demo

### Option 1: Using Virtual Environment
```powershell
# Activate venv
& d:/fed/federated-health-risk-mlops/.venv/Scripts/Activate.ps1

# Run demo
python demo_mlops.py
```

### Option 2: Direct Python
```powershell
python demo_mlops.py
```

## What You'll See

The demo will:
1. Start Prometheus metrics server on port 8081
2. Create an MLflow experiment
3. Simulate 10 rounds of federated learning
4. Log metrics to MLflow after each round
5. Expose real-time metrics via Prometheus

## Access the Results

### MLflow UI
Open browser: http://localhost:5000

You'll see:
- Experiment runs with parameters
- Metrics graphs (loss, accuracy) over rounds
- Run comparison features

### Prometheus Metrics
Open browser: http://localhost:8081/metrics

You'll see metrics like:
- `fl_training_loss`
- `fl_validation_loss`
- `fl_accuracy`
- `fl_num_clients`
- `fl_total_rounds`

## Sample Output

```
🚀 Starting Prometheus metrics server on port 8081
   Metrics available at: http://localhost:8081/metrics

📊 MLflow tracking URI: http://localhost:5000
   MLflow UI: http://localhost:5000

============================================================
Starting Federated Learning Demo
============================================================

────────────────────────────────────────────────────────────
⚡ ROUND 1/10
────────────────────────────────────────────────────────────
   Client 1: Training...
   Client 1: Loss=2.3845, Acc=0.3421, Samples=1053
   Client 2: Training...
   Client 2: Loss=2.4521, Acc=0.3189, Samples=987
   Client 3: Training...
   Client 3: Loss=2.3124, Acc=0.3654, Samples=1121

   📈 AGGREGATED RESULTS:
      Train Loss:  2.3789
      Val Loss:    2.6168
      Accuracy:    0.3421 (34.21%)
      Duration:    1.52s
```

## Stop the Demo

Press `Ctrl+C` to stop the metrics server.

## Tips

- Run multiple times to create multiple experiment runs in MLflow
- Compare runs in MLflow UI
- Use this as a template for real federated learning integration
