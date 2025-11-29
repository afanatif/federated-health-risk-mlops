# Federated Learning Deployment Guide

## Overview

This guide covers deploying the Federated  Learning system using Docker Compose (local) or Kubernetes (production).

## Prerequisites

### Required Software
- **Docker Desktop**: v29.0.1+ with Docker Compose v2.40.3+
- **Minikube** (for Kubernetes): Already installed
- **kubectl**: Comes with Minikube
- **Python**: 3.10+ (for local development)

### System Requirements
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk**: 50GB free space for datasets and models
- **CPU**: 4+ cores recommended

## Quick Start

### Option 1: Docker Compose (Local Development)

Best for testing and development on a single machine.

```powershell
# 1. Build Docker images
.\scripts\build_images.ps1

# 2. Start all services
docker-compose up -d

# 3. View logs
docker-compose logs -f server

# 4. Access services
# MLflow UI: http://localhost:5000
# Prometheus metrics: http://localhost:8081/metrics

# 5. Stop services
docker-compose down
```

### Option 2: Kubernetes with Minikube (Production-like)

Best for production deployment and testing scalability.

```powershell
# 1. Install Minikube (if not done)
.\minikube-installer.exe

# 2. Start Minikube
minikube start --driver=docker --memory=8192 --cpus=4

# 3. Build images
.\scripts\build_images.ps1

# 4. Deploy to Kubernetes
.\scripts\deploy_k8s.ps1

# 5. Access services
# MLflow UI: http://$(minikube ip):30500
# Flower Server: $(minikube ip):30080
# Prometheus: http://$(minikube ip):30081/metrics

# Or use minikube service
minikube service mlflow-service -n federated-learning
```

## Architecture

### Components

1. **MLflow Tracking Server**
   - Tracks experiments and metrics
   - Stores model artifacts
   - Web UI on port 5000

2. **Federated Server**
   - Coordinates federated training
   - Aggregates client models (FedAvg)
   - Exposes Flower gRPC on port 8080
   - Prometheus metrics on port 8081

3. **Client Nodes** (3 replicas)
   - Train on local data
   - Send updates to server
   - No data leaves client

### Data Flow

```
Client 1 ──┐
           ├──> FL Server ──> MLflow ──> Model Registry
Client 2 ──┤        │
Client 3 ──┘        └──> Prometheus ──> Grafana
```

## Configuration

### Environment Variables

Edit `k8s/configmap.yaml` or `docker-compose.yml`:

```yaml
FL_ROUNDS: "10"              # Number of training rounds
FL_MIN_CLIENTS: "3"          # Minimum clients to start
MODEL_SIZE: "n"              # YOLOv8 size (n/s/m/l/x)
NUM_CLASSES: "7"             # Number of object classes
MLFLOW_TRACKING_URI: "..."   # MLflow server URL
```

### Resource Limits

For production, adjust in `k8s/server-deployment.yaml`:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Monitoring

### MLflow UI

Access experiment tracking:
- **Docker Compose**: http://localhost:5000
- **Kubernetes**: http://$(minikube ip):30500

View:
- Training metrics per round
- Model versions
- Hyperparameters
- Artifacts

### Prometheus Metrics

Scrape metrics from:
- **Docker Compose**: http://localhost:8081/metrics
- **Kubernetes**: http://$(minikube ip):30081/metrics

Available metrics:
- `fl_train_loss` - Current training loss
- `fl_accuracy` - Current accuracy
- `fl_rounds_completed_total` - Total rounds completed
- `fl_active_clients` - Number of active clients
- `fl_round_duration_seconds` - Training time histogram

### Logs

**Docker Compose:**
```powershell
docker-compose logs -f server    # Server logs
docker-compose logs -f client1   # Client logs
```

**Kubernetes:**
```powershell
kubectl logs -f deployment/fl-server -n federated-learning
kubectl logs -f fl-client-0 -n federated-learning
```

## Troubleshooting

### Common Issues

**1. Minikube won't start**
```powershell
# Check virtualization
minikube start --driver=docker --alsologtostderr -v=7

# Try with more resources
minikube start --memory=8192 --cpus=4
```

**2. Images not found in Kubernetes**
```powershell
# Load images into Minikube
.\scripts\build_images.ps1
minikube image load federated-learning/server:latest
minikube image load federated-learning/client:latest
minikube image load federated-learning/mlflow:latest
```

**3. PVC pending**
```powershell
# Check storage class
kubectl get storageclass

# Minikube uses 'standard' by default
# If missing, create it:
minikube addons enable storage-provisioner
```

**4. Clients can't connect to server**
```powershell
# Check server is running
kubectl get pods -n federated-learning

# Check server logs
kubectl logs deployment/fl-server -n federated-learning

# Verify service
kubectl get svc -n federated-learning
```

**5. MLflow UI not accessible**
```powershell
# Port forward to localhost
kubectl port-forward -n federated-learning svc/mlflow-service 5000:5000

# Or use minikube tunnel
minikube tunnel
```

## Scaling

### Add More Clients

**Docker Compose:**
Edit `docker-compose.yml` and add more client services.

**Kubernetes:**
```powershell
kubectl scale statefulset fl-client --replicas=5 -n federated-learning
```

### Production Deployment

For real production (GKE, EKS, AKS):

1. **Push images to registry:**
   ```powershell
   $env:REGISTRY="gcr.io/your-project"
   .\scripts\build_images.ps1 -Push
   ```

2. **Update image references** in K8s manifests

3. **Configure persistent storage** for your cloud provider

4. **Set up ingress** for external access

5. **Configure secrets** for sensitive data

## Cleanup

### Docker Compose
```powershell
docker-compose down -v  # -v removes volumes
```

### Kubernetes
```powershell
kubectl delete namespace federated-learning
# Or
minikube delete  # Completely remove cluster
```

## Next Steps

- Set up Grafana for visualization: See [MONITORING.md](MONITORING.md)
- Configure model versioning: See [MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md)
- Enable automatic re-training triggers
- Set up CI/CD pipelines
