# Model Management and Re-training Guide

## Overview

This guide explains how to manage model versions, promote models through stages, and configure automated re-training.

## Model Versioning with MLflow Model Registry

### Model Lifecycle Stages

Models progress through these stages:
1. **None**: New model, not yet reviewed
2. **Staging**: Testing in staging environment
3. **Production**: Currently deployed model
4. **Archived**: Old models, kept for reference

### Using the Model Manager

```python
from server.model_manager import ModelManager

# Initialize
manager = ModelManager(
    tracking_uri="http://localhost:5000",
    registered_model_name="fl-global-model"
)

# Save new version
version = manager.save_model_version(
    model_path="server/checkpoints/global_round_10.pt",
    run_id="abc123",
    metrics={"accuracy": 0.85, "loss": 0.42},
    tags={"round": "10", "clients": "3"}
)
print(f"Saved model version: {version}")

# Promote to staging
manager.promote_to_staging(version)

# After testing, promote to production
manager.promote_to_production(version)

# List all versions
versions = manager.list_all_versions()
for v in versions:
    print(f"v{v['version']}: {v['stage']} - {v['created']}")
```

### Model Comparison

```python
# Compare two versions
comparison = manager.compare_versions("1", "2")

print("Version 1:")
print(f"  Accuracy: {comparison['version1']['metrics']['accuracy']}")
print(f"  Stage: {comparison['version1']['stage']}")

print("\nVersion 2:")
print(f"  Accuracy: {comparison['version2']['metrics']['accuracy']}")
print(f"  Stage: {comparison['version2']['stage']}")
```

### Loading Production Model

```python
from models.model import get_model

# Get model class
model = get_model(model_size='n', num_classes=7)

# Load production weights
production_model = manager.load_production_model(
    model_class=lambda: get_model(model_size='n', num_classes=7),
    device='cpu'
)

if production_model:
    print("✅ Production model loaded")
    # Use for inference
    predictions = production_model.predict(images)
```

## Automated Re-training

### Re-training Triggers

The system supports three trigger types:

1. **Performance Degradation**: Triggered when accuracy drops
2. **Scheduled**: Periodic re-training (e.g., weekly)
3. **Manual**: Triggered by API call or admin

### Setting Up Re-training Manager

```python
from server.retraining_trigger import RetrainingManager

# Initialize
retrain_mgr = RetrainingManager(
    accuracy_threshold=0.05,  # Trigger if accuracy drops 5%
    check_interval=300,       # Check every 5 minutes
    min_rounds_between_retrain=10  # Wait at least 10 rounds
)

# Define what happens when retraining is triggered
def start_federated_training(trigger):
    print(f"🔄 Re-training triggered: {trigger.reason}")
    # Start new federated learning round
    # ... your training logic here ...

retrain_mgr.set_retrain_callback(start_federated_training)

# Schedule periodic retraining (every 7 days)
retrain_mgr.schedule_periodic_retrain(interval_days=7)

# Start monitoring
retrain_mgr.start_monitoring()

# Update metrics after each round
retrain_mgr.update_metrics(accuracy=0.82, round_num=15)
```

### Manual Trigger via API

The re-training manager includes a REST API:

```python
from server.retraining_trigger import create_trigger_api, RetrainingManager
from flask import Flask

retrain_mgr = RetrainingManager()
app = create_trigger_api(retrain_mgr)

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

**Trigger via HTTP:**
```powershell
# Trigger retraining
curl -X POST http://localhost:5001/trigger_retrain `
  -H "Content-Type: application/json" `
  -d '{"reason": "Manual trigger for testing"}'

# Get trigger history
curl http://localhost:5001/trigger_history

# Get current metrics
curl http://localhost:5001/metrics
```

### Kubernetes CronJob for Scheduled Re-training

```yaml
# k8s/retraining-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: fl-retrain
  namespace: federated-learning
spec:
  # Run every Sunday at 2 AM
  schedule: "0 2 * * 0"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: retrain-trigger
            image: curlimages/curl:latest
            command:
            - curl
            - -X
            - POST
            - http://fl-server:5001/trigger_retrain
            - -H
            - "Content-Type: application/json"
            - -d
            - '{"reason": "Scheduled weekly retraining"}'
          restartPolicy: OnFailure
```

Apply:
```powershell
kubectl apply -f k8s/retraining-cronjob.yaml
```

## Model Deployment Workflow

### End-to-End Process

```
1. Train → 2. Validate → 3. Stage → 4. Test → 5. Production
```

**1. Training Phase:**
- Federated learning completes N rounds
- Model checkpoint saved to MLflow
- Metrics logged

**2. Validation Phase:**
```python
# Load latest model
latest_version = manager.get_latest_version(stage="None")

# Run validation
accuracy = validate_model(model, validation_dataset)

if accuracy > 0.80:
    manager.promote_to_staging(latest_version)
    print("✅ Model promoted to staging")
else:
    print("❌ Model accuracy too low, not promoting")
```

**3. Staging Phase:**
```python
# Test in staging environment
staging_version = manager.get_latest_version(stage="Staging")
staging_model = manager.load_production_model(...)

# Run integration tests
# Check for regression
# A/B testing

if tests_passed:
    manager.promote_to_production(staging_version)
```

**4. Production Deployment:**
```python
# Automatic rollback on failure
try:
    production_model = manager.load_production_model(...)
    deploy_to_production(production_model)
except Exception as e:
    # Rollback to previous version
    previous_version = get_previous_version()
    manager.promote_to_production(previous_version)
    logger.error(f"Deployment failed, rolled back: {e}")
```

## Model Monitoring

### Track Model Performance

```python
from scripts.experiment_tracking import FederatedLearningTracker

tracker = FederatedLearningTracker()
tracker.start_run(run_name="production_inference_v1")

# Log inference metrics
tracker.log_metrics({
    "inference_accuracy": 0.87,
    "avg_latency_ms": 45,
    "throughput_qps": 100
}, step=1)

# Log predictions for analysis
tracker.log_artifact("predictions.json")

tracker.end_run()
```

### Data Drift Detection

Monitor for changes in input distribution:

```python
def check_data_drift(current_data, reference_data):
    from scipy.stats import ks_2samp
    
    # Compare distributions
    statistic, pvalue = ks_2samp(current_data, reference_data)
    
    if pvalue < 0.05:
        print("⚠️ Data drift detected!")
        # Trigger retraining
        retrain_mgr.manual_trigger(
            reason=f"Data drift detected (p-value: {pvalue})"
        )
    
    return pvalue
```

## Best Practices

### Versioning
1. **Always tag versions** with metadata (round, accuracy, timestamp)
2. **Keep production history** - don't delete old production models
3. **Document changes** in model architecture or training process

### Testing
1. **Validate on holdout set** before promoting
2. **Run regression tests** to ensure no degradation
3. **A/B test** new models before full rollout

### Re-training
1. **Set conservative thresholds** to avoid too-frequent retraining
2. **Monitor trigger frequency** in production
3. **Manual review** before promoting retrained models
4. **Keep retraining data** for reproducibility

### Rollback
1. **Always keep previous production model** available
2. **Test rollback procedure** regularly
3. **Monitor metrics** after deployment
4. **Automate rollback** on critical metric degradation

## Troubleshooting

### Model Registry Issues

**Model not found:**
```python
# Check if model exists
try:
    model = manager.client.get_registered_model("fl-global-model")
    print(f"Model exists: {model.name}")
except:
    print("Model not found - will be created on first save")
```

**Version mismatch:**
```python
# Verify version
version_info = manager.client.get_model_version(
    "fl-global-model", 
    version="1"
)
print(f"Source: {version_info.source}")
print(f"Run ID: {version_info.run_id}")
```

### Re-training Not Triggering

1. **Check monitoring is running:**
```python
print(f"Monitoring: {retrain_mgr.monitoring}")
```

2. **Verify callback is set:**
```python
if retrain_mgr.retrain_callback is None:
    retrain_mgr.set_retrain_callback(your_function)
```

3. **Check thresholds:**
```python
print(f"Best: {retrain_mgr.best_accuracy}")
print(f"Current: {retrain_mgr.current_accuracy}")
print(f"Threshold: {retrain_mgr.accuracy_threshold}")
```

### Performance Issues

**Slow model loading:**
- Use local artifact storage instead of S3
- Cache frequently-used models
- Optimize model size

**High storage usage:**
- Archive old models
- Set retention policy in MLflow
- Use model compression

## Example: Complete Workflow

```python
from server.model_manager import ModelManager
from server.retraining_trigger import RetrainingManager
from scripts.experiment_tracking import FederatedLearningTracker

# 1. Initialize components
manager = ModelManager()
retrain_mgr = RetrainingManager()
tracker = FederatedLearningTracker()

# 2. Start training run
tracker.start_run("fl_training_round_1")
tracker.log_params({
    "rounds": 10,
    "clients": 3,
    "model": "yolov8n"
})

# 3. Train model (federated learning)
# ... training happens ...

# 4. Save model version
version = manager.save_model_version(
    model_path="server/checkpoints/global_round_10.pt",
    run_id=tracker.run.info.run_id,
    metrics={"accuracy": 0.85},
    tags={"round": "10"}
)

# 5. Validate and promote
if accuracy > 0.80:
    manager.promote_to_staging(version)
    # Test in staging...
    manager.promote_to_production(version)

# 6. Update retraining manager
retrain_mgr.update_metrics(accuracy=0.85, round_num=10)

# 7. Set up automated retraining
retrain_mgr.set_retrain_callback(start_new_training)
retrain_mgr.schedule_periodic_retrain(interval_days=7)
retrain_mgr.start_monitoring()

# 8. End tracking
tracker.end_run()
```
