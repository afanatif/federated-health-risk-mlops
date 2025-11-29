# Complete Deployment Script for Federated Learning MLOps System
# This script orchestrates the entire deployment process
# Usage: .\deploy_complete.ps1

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = $PSScriptRoot | Split-Path -Parent
Set-Location $PROJECT_ROOT

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Federated Learning Complete Deployment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify Docker is running
Write-Host "Step 1: Checking Docker status..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host  "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Check if Minikube is already running
Write-Host "Step 2: Checking Minikube status..." -ForegroundColor Yellow
$minikubeRunning = $false
try {
    $status = minikube status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Minikube is already running" -ForegroundColor Green
        $minikubeRunning = $true
    }
}
catch {
    Write-Host "Minikube not running yet" -ForegroundColor Gray
}

if (-not $minikubeRunning) {
    Write-Host "Starting Minikube (this may take a few minutes)..." -ForegroundColor Yellow
    minikube start --driver=docker --memory=3072 --cpus=2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to start Minikube" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Minikube started successfully" -ForegroundColor Green
}
Write-Host ""

# Step 3: Load Docker images into Minikube
Write-Host "Step 3: Loading Docker images into Minikube..." -ForegroundColor Yellow
Write-Host "  Loading server image..." -ForegroundColor Cyan
minikube image load federated-learning/server:latest
Write-Host "  Loading client image..." -ForegroundColor Cyan
minikube image load federated-learning/client:latest
Write-Host "  Loading MLflow image..." -ForegroundColor Cyan
minikube image load federated-learning/mlflow:latest
Write-Host "✅ All images loaded into Minikube" -ForegroundColor Green
Write-Host ""

# Step 4: Deploy Kubernetes resources
Write-Host "Step 4: Deploying to Kubernetes..." -ForegroundColor Yellow

Write-Host "  📁 Creating namespace..." -ForegroundColor Cyan
kubectl apply -f k8s/namespace.yaml

Write-Host "  ⚙️  Creating ConfigMap..." -ForegroundColor Cyan
kubectl apply -f k8s/configmap.yaml

Write-Host "  💾 Creating PersistentVolumeClaims..." -ForegroundColor Cyan
kubectl apply -f k8s/pvc.yaml

Write-Host "  📊 Deploying MLflow..." -ForegroundColor Cyan
kubectl apply -f k8s/mlflow-deployment.yaml

Write-Host "  🖥️  Deploying FL Server..." -ForegroundColor Cyan
kubectl apply -f k8s/server-deployment.yaml

Write-Host "  👥 Deploying FL Clients..." -ForegroundColor Cyan
kubectl apply -f k8s/client-deployment.yaml

Write-Host "  📈 Deploying Prometheus..." -ForegroundColor Cyan
kubectl apply -f k8s/prometheus-deployment.yaml

Write-Host "  📊 Deploying Grafana..." -ForegroundColor Cyan
kubectl apply -f k8s/grafana-deployment.yaml

Write-Host ""
Write-Host "✅ All resources deployed" -ForegroundColor Green
Write-Host ""

# Step 5: Wait for deployments
Write-Host "Step 5: Waiting for deployments to be ready..." -ForegroundColor Yellow
Write-Host "  Waiting for MLflow..." -ForegroundColor Cyan
kubectl wait --for=condition=available --timeout=300s deployment/mlflow-server -n federated-learning 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ MLflow ready" -ForegroundColor Green
}

Write-Host "  Waiting for FL Server..." -ForegroundColor Cyan
kubectl wait --for=condition=available --timeout=300s deployment/fl-server -n federated-learning 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ FL Server ready" -ForegroundColor Green
}

Write-Host "  Waiting for Prometheus..." -ForegroundColor Cyan
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n federated-learning 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Prometheus ready" -ForegroundColor Green
}

Write-Host "  Waiting for Grafana..." -ForegroundColor Cyan
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n federated-learning 2>$null  
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Grafana ready" -ForegroundColor Green
}

Write-Host "  Waiting for FL Clients..." -ForegroundColor Cyan
Start-Sleep-Seconds 10  # Give clients time to start
kubectl wait --for=condition=ready --timeout=300s pod -l app=fl-client -n federated-learning 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ FL Clients ready" -ForegroundColor Green
}

Write-Host ""
Write-Host "✅ All deployments are ready!" -ForegroundColor Green
Write-Host ""

# Step 6: Display access information
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🎉 Deployment Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$minikubeIP = minikube ip

Write-Host "📡 Service Endpoints:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  MLflow UI:" -ForegroundColor Cyan
Write-Host "    http://${minikubeIP}:30500" -ForegroundColor Green
Write-Host ""
Write-Host "  Prometheus UI:" -ForegroundColor Cyan
Write-Host "    http://${minikubeIP}:30090" -ForegroundColor Green
Write-Host ""
Write-Host "  Grafana Dashboard:" -ForegroundColor Cyan
Write-Host "    http://${minikubeIP}:30300" -ForegroundColor Green
Write-Host "    Default login: admin / admin" -ForegroundColor Gray
Write-Host ""
Write-Host "  Server Metrics:" -ForegroundColor Cyan
Write-Host "    http://${minikubeIP}:30081/metrics" -ForegroundColor Green
Write-Host ""

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "📝 Useful Commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  View all pods:" -ForegroundColor Cyan
Write-Host "    kubectl get pods -n federated-learning" -ForegroundColor Gray
Write-Host ""
Write-Host "  View server logs:" -ForegroundColor Cyan
Write-Host "    kubectl logs -f deployment/fl-server -n federated-learning" -ForegroundColor Gray
Write-Host ""
Write-Host "  View client logs (replace N with 0, 1, or 2):" -ForegroundColor Cyan
Write-Host "    kubectl logs -f fl-client-N -n federated-learning" -ForegroundColor Gray
Write-Host ""
Write-Host "  Open MLflow UI in browser:" -ForegroundColor Cyan
Write-Host "    minikube service mlflow-service -n federated-learning" -ForegroundColor Gray
Write-Host ""
Write-Host "  Delete all resources:" -ForegroundColor Cyan
Write-Host "    kubectl delete namespace federated-learning" -ForegroundColor Gray
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
