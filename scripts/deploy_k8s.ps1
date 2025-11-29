# Deploy Federated Learning System to Kubernetes
# PowerShell version for Windows with Minikube
# Usage: .\scripts\deploy_k8s.ps1

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $PROJECT_ROOT

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Deploying Federated Learning to Kubernetes" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if minikube is running
Write-Host "Checking Minikube status..." -ForegroundColor Yellow
try {
    $minikubeStatus = minikube status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Minikube is not running. Starting Minikube..." -ForegroundColor Yellow
        minikube start --driver=docker
        Write-Host "✅ Minikube started" -ForegroundColor Green
    }
    else {
        Write-Host "✅ Minikube is running" -ForegroundColor Green
    }
}
catch {
    Write-Host "❌ Minikube not found. Please install Minikube first." -ForegroundColor Red
    Write-Host "   Run: .\minikube-installer.exe (already downloaded)" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Load Docker images into Minikube
Write-Host "Loading Docker images into Minikube..." -ForegroundColor Yellow
minikube image load federated-learning/server:latest
minikube image load federated-learning/client:latest
minikube image load federated-learning/mlflow:latest
Write-Host "✅ Images loaded into Minikube" -ForegroundColor Green
Write-Host ""

# Apply Kubernetes manifests
Write-Host "Applying Kubernetes manifests..." -ForegroundColor Yellow

Write-Host "  📁 Creating namespace..." -ForegroundColor Cyan
kubectl apply -f k8s/namespace.yaml

Write-Host "  ⚙️  Creating ConfigMap..." -ForegroundColor Cyan
kubectl apply -f k8s/configmap.yaml

Write-Host "  💾 Creating PersistentVolumeClaims..." -ForegroundColor Cyan
kubectl apply -f k8s/pvc.yaml

Write-Host "  📊 Deploying MLflow..." -ForegroundColor Cyan
kubectl apply -f k8s/mlflow-deployment.yaml

Write-Host "  🖥️  Deploying Server..." -ForegroundColor Cyan
kubectl apply -f k8s/server-deployment.yaml

Write-Host "  👥 Deploying Clients..." -ForegroundColor Cyan
kubectl apply -f k8s/client-deployment.yaml

Write-Host ""
Write-Host "✅ All manifests applied" -ForegroundColor Green
Write-Host ""

# Wait for deployments
Write-Host "Waiting for deployments to be ready..." -ForegroundColor Yellow
Write-Host "  Waiting for MLflow..."
kubectl wait --for=condition=available --timeout=300s deployment/mlflow-server -n federated-learning

Write-Host "  Waiting for Server..."
kubectl wait --for=condition=available --timeout=300s deployment/fl-server -n federated-learning

Write-Host "  Waiting for Clients..."
kubectl wait --for=condition=ready --timeout=300s pod -l app=fl-client -n federated-learning

Write-Host ""
Write-Host "✅ All deployments ready!" -ForegroundColor Green
Write-Host ""

# Get service URLs
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Service Endpoints" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$minikubeIP = minikube ip

Write-Host "MLflow UI:" -ForegroundColor Yellow
Write-Host "  http://${minikubeIP}:30500" -ForegroundColor Green
Write-Host ""

Write-Host "Flower Server (gRPC):" -ForegroundColor Yellow
Write-Host "  ${minikubeIP}:30080" -ForegroundColor Green
Write-Host ""

Write-Host "Prometheus Metrics:" -ForegroundColor Yellow
Write-Host "  http://${minikubeIP}:30081/metrics" -ForegroundColor Green
Write-Host ""

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  View pods:       kubectl get pods -n federated-learning"
Write-Host "  View logs:       kubectl logs -f deployment/fl-server -n federated-learning"
Write-Host "  Open MLflow UI:  minikube service mlflow-service -n federated-learning"
Write-Host "  Delete all:      kubectl delete namespace federated-learning"
Write-Host ""
