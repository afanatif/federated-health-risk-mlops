# Build Docker images for federated learning system
# PowerShell version for Windows
# Usage: .\scripts\build_images.ps1 [-Push]

param(
    [switch]$Push = $false
)

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $PROJECT_ROOT

$VERSION = if ($env:VERSION) { $env:VERSION } else { "latest" }
$REGISTRY = if ($env:REGISTRY) { $env:REGISTRY } else { "federated-learning" }

if ($Push) {
    Write-Host "Will push images to registry after building" -ForegroundColor Yellow
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Building Docker Images for Federated Learning" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Version: $VERSION"
Write-Host "Registry: $REGISTRY"
Write-Host ""

# Build server image
Write-Host "📦 Building server image..." -ForegroundColor Green
docker build -f docker/Dockerfile.server -t "${REGISTRY}/server:${VERSION}" .
Write-Host "✅ Server image built: ${REGISTRY}/server:${VERSION}" -ForegroundColor Green
Write-Host ""

# Build client image
Write-Host "📦 Building client image..." -ForegroundColor Green
docker build -f docker/Dockerfile.client -t "${REGISTRY}/client:${VERSION}" .
Write-Host "✅ Client image built: ${REGISTRY}/client:${VERSION}" -ForegroundColor Green
Write-Host ""

# Build MLflow image
Write-Host "📦 Building MLflow image..." -ForegroundColor Green
docker build -f docker/Dockerfile.mlflow -t "${REGISTRY}/mlflow:${VERSION}" .
Write-Host "✅ MLflow image built: ${REGISTRY}/mlflow:${VERSION}" -ForegroundColor Green
Write-Host ""

# Tag as latest
Write-Host "📌 Tagging images as latest..." -ForegroundColor Yellow
docker tag "${REGISTRY}/server:${VERSION}" "${REGISTRY}/server:latest"
docker tag "${REGISTRY}/client:${VERSION}" "${REGISTRY}/client:latest"
docker tag "${REGISTRY}/mlflow:${VERSION}" "${REGISTRY}/mlflow:latest"
Write-Host "✅ Images tagged as latest" -ForegroundColor Green
Write-Host ""

# Push images if requested
if ($Push) {
    Write-Host "🚀 Pushing images to registry..." -ForegroundColor Yellow
    docker push "${REGISTRY}/server:${VERSION}"
    docker push "${REGISTRY}/server:latest"
    docker push "${REGISTRY}/client:${VERSION}"
    docker push "${REGISTRY}/client:latest"
    docker push "${REGISTRY}/mlflow:${VERSION}"
    docker push "${REGISTRY}/mlflow:latest"
    Write-Host "✅ Images pushed successfully" -ForegroundColor Green
    Write-Host ""
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Images built:"
Write-Host "  - ${REGISTRY}/server:${VERSION}"
Write-Host "  - ${REGISTRY}/client:${VERSION}"
Write-Host "  - ${REGISTRY}/mlflow:${VERSION}"
Write-Host ""
Write-Host "To run locally with Docker Compose:"
Write-Host "  docker-compose up -d" -ForegroundColor Yellow
Write-Host ""
Write-Host "To deploy to Kubernetes:"
Write-Host "  minikube start"
Write-Host "  kubectl apply -f k8s/" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
