# Monitoring and Observability Guide

## Overview

This guide explains how to monitor your federated learning system using MLflow and Prometheus/Grafana.

## MLflow Tracking

### Accessing the UI

- **Local (Docker Compose)**: http://localhost:5000
- **Kubernetes**: http://$(minikube ip):30500

### What's Tracked

#### Experiments
Each federated learning run creates an experiment in MLflow:
- Unique run ID
- Start/end time
- Status (running, finished, failed)

#### Parameters
- `num_rounds`: Training rounds
- `num_clients`: Number of clients
- `strategy`: Aggregation strategy (FedAvg)
- `model_name`: YOLOv8n
- `learning_rate`, `batch_size`, etc.

#### Metrics (Per Round)
- `train_loss`: Training loss after aggregation
- `eval_loss`: Evaluation loss
- `accuracy`: Model accuracy
- `num_clients`: Participating clients

#### Artifacts
- Model checkpoints (`.pt` files)
- Training logs
- Configuration files

### Comparing Runs

```python
from scripts.experiment_tracking import FederatedLearningTracker

tracker = FederatedLearningTracker()

# Get best run
best_run = tracker.get_best_run(metric='accuracy', mode='max')

# Compare multiple runs
comparison = tracker.compare_runs(['run_id_1', 'run_id_2'])
```

## Prometheus Metrics

### Metrics Endpoint

- **Local**: http://localhost:8081/metrics
- **Kubernetes**: http://$(minikube ip):30081/metrics

### Available Metrics

#### Counters
- `fl_rounds_completed_total`: Total completed rounds
- `fl_rounds_failed_total`: Total failed rounds

#### Gauges
- `fl_train_loss`: Current training loss
- `fl_eval_loss`: Current evaluation loss
- `fl_accuracy`: Current accuracy
- `fl_best_accuracy`: Best accuracy achieved
- `fl_active_clients`: Active clients in current round
- `fl_total_clients`: Total registered clients
- `fl_model_parameters`: Number of model parameters
- `fl_samples_per_round`: Training samples per round

#### Histograms
- `fl_round_duration_seconds`: Training time distribution

### Setting Up Prometheus

**1. Create Prometheus ConfigMap:**

```yaml
# k8s/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: federated-learning
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'fl-server'
        static_configs:
          - targets: ['fl-server:8081']
```

**2. Deploy Prometheus:**

```yaml
# k8s/prometheus-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: federated-learning
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: federated-learning
spec:
  type: NodePort
  ports:
  - port: 9090
    targetPort: 9090
    nodePort: 30090
  selector:
    app: prometheus
```

**3. Apply:**
```powershell
kubectl apply -f k8s/prometheus-config.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
```

**4. Access:**
```powershell
# Kubernetes
minikube service prometheus -n federated-learning

# Or port-forward
kubectl port-forward -n federated-learning svc/prometheus 9090:9090
```

## Setting Up Grafana

### Deploy Grafana

**1. Create deployment:**

```yaml
# k8s/grafana-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: federated-learning
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin"
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: federated-learning
spec:
  type: NodePort
  ports:
  - port: 3000
    targetPort: 3000
    nodePort: 30030
  selector:
    app: grafana
```

**2. Apply:**
```powershell
kubectl apply -f k8s/grafana-deployment.yaml
```

**3. Access:**
- URL: http://$(minikube ip):30030
- Username: `admin`
- Password: `admin`

### Configure Datasource

1. Go to **Configuration** → **Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. Set URL: `http://prometheus:9090`
5. Click **Save & Test**

### Import Dashboard

Create a dashboard with these panels:

#### Training Loss Panel
```
Query: fl_train_loss
Type: Graph
```

#### Accuracy Panel
```
Query: fl_accuracy
Type: Graph
```

#### Active Clients Panel
```
Query: fl_active_clients
Type: Stat
```

#### Round Duration Panel
```
Query: histogram_quantile(0.95, fl_round_duration_seconds_bucket)
Type: Graph
```

### Pre-built Dashboard JSON

```json
{
  "dashboard": {
    "title": "Federated Learning Monitoring",
    "panels": [
      {
        "title": "Training Loss",
        "targets": [{"expr": "fl_train_loss"}]
      },
      {
        "title": "Accuracy",
        "targets": [{"expr": "fl_accuracy"}]
      },
      {
        "title": "Active Clients",  
        "targets": [{"expr": "fl_active_clients"}]
      }
    ]
  }
}
```

Import via: **Dashboards** → **Import** → **Paste JSON**

## Alerting

### Prometheus Alert Rules

Create alert rules in `prometheus.yml`:

```yaml
rule_files:
  - /etc/prometheus/alerts.yml

# alerts.yml
groups:
  - name: federated_learning
    rules:
    - alert: AccuracyDropped
      expr: fl_accuracy < 0.5
      for: 5m
      annotations:
        summary: "Model accuracy dropped below 50%"
        
    - alert: NoActiveClients
      expr: fl_active_clients == 0
      for: 2m
      annotations:
        summary: "No active clients connected"
        
    - alert: TrainingStalled
      expr: rate(fl_rounds_completed_total[10m]) == 0
      for: 10m
      annotations:
        summary: "No rounds completed in 10 minutes"
```

## Logs Analysis

### Centralized Logging

For production, use ELK stack or cloud logging:

**Fluentd DaemonSet:**
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: federated-learning
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
```

### Viewing Logs

**Docker Compose:**
```powershell
docker-compose logs -f --tail=100 server
```

**Kubernetes:**
```powershell
# Real-time server logs
kubectl logs -f deployment/fl-server -n federated-learning

# Last 100 lines from all clients
kubectl logs --tail=100 -l app=fl-client -n federated-learning
```

## Performance Monitoring

### Key Metrics to Watch

1. **Training Progress**
   - Loss decreasing?
   - Accuracy improving?
   - Convergence rate

2. **System Health**
   - CPU/Memory usage
   - Network bandwidth
   - Storage capacity

3. **Client Participation**
   - Number of active clients
   - Client failures
   - Data distribution

### Kubernetes Resource Metrics

```powershell
# Pod resource usage
kubectl top pods -n federated-learning

# Node resource usage
kubectl top nodes
```

## Best Practices

1. **Set up alerts** for critical metrics
2. **Monitor trends** over time, not just current values
3. **Track model performance** on validation set
4. **Log everything** in production
5. **Use dashboards** for at-a-glance status
6. **Archive logs** and metrics for analysis
7. **Set retention policies** for storage management

## Troubleshooting

### Metrics not appearing
- Check Prometheus targets: http://$(minikube ip):30090/targets
- Verify server metrics endpoint: curl http://localhost:8081/metrics
- Check firewall rules

### MLflow UI slow
- Clear old runs: mlflow gc
- Use database backend instead of file store
- Increase resource limits

### Grafana not showing data
- Verify Prometheus datasource connection
- Check time range in queries
- Validate PromQL syntax
