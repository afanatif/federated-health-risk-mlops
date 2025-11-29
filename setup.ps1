# Setup Verification Script
# Run this after cloning from GitHub to ensure everything works

param(
    [switch]$SkipBuild
)

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  Federated Learning MLOps - Setup Verification" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker
Write-Host "[1/5] Checking Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker installed: $dockerVersion" -ForegroundColor Green
    
    docker info | Out-Null
    Write-Host "✅ Docker daemon is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not available!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop:" -ForegroundColor Yellow
    Write-Host "  Windows/Mac: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Write-Host "  Linux: sudo apt-get install docker-compose" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check required files
Write-Host ""
Write-Host "[2/5] Checking required files..." -ForegroundColor Yellow
$requiredFiles = @(
    "docker-compose.yml",
    "docker/Dockerfile.server",
    "docker/Dockerfile.client",
    "docker/Dockerfile.mlflow",
    "demos/federated_3_rounds.py"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ Found: $file" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host ""
    Write-Host "❌ Some required files are missing!" -ForegroundColor Red
    Write-Host "   Make sure you cloned the complete repository" -ForegroundColor Yellow
    exit 1
}

# Step 3: Build and start services
Write-Host ""
if ($SkipBuild) {
    Write-Host "[3/5] Starting services (skipping build)..." -ForegroundColor Yellow
    docker-compose up -d
}
else {
    Write-Host "[3/5] Building and starting services..." -ForegroundColor Yellow
    Write-Host "   This may take 5-10 minutes on first run..." -ForegroundColor Gray
    docker-compose up -d --build
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start services!" -ForegroundColor Red
    Write-Host "   Check logs with: docker-compose logs" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Services started" -ForegroundColor Green

# Step 4: Wait for services to be ready
Write-Host ""
Write-Host "[4/5] Waiting for services to be ready..." -ForegroundColor Yellow
Write-Host "   Waiting 30 seconds..." -ForegroundColor Gray
Start-Sleep -Seconds 30

# Check MLflow
$mlflowReady = $false
for ($i = 1; $i -le 10; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 2 -UseBasicParsing
        $mlflowReady = $true
        break
    }
    catch {
        Write-Host "   Attempt $i/10: MLflow not ready yet..." -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
}

if ($mlflowReady) {
    Write-Host "✅ MLflow is accessible at http://localhost:5000" -ForegroundColor Green
}
else {
    Write-Host "⚠️  MLflow may not be ready yet" -ForegroundColor Yellow
    Write-Host "   Check with: docker logs fl-mlflow" -ForegroundColor Gray
}

# Step 5: Run demo
Write-Host ""
Write-Host "[5/5] Running demo (3 federated rounds)..." -ForegroundColor Yellow

try {
    # Copy demo script
    docker cp demos/federated_3_rounds.py fl-mlflow:/tmp/ | Out-Null
    
    # Execute demo
    docker exec fl-mlflow python /tmp/federated_3_rounds.py
    
    Write-Host "✅ Demo completed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "⚠️  Demo execution had issues" -ForegroundColor Yellow
    Write-Host "   You can run it manually later" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. View MLflow Dashboard:" -ForegroundColor White
Write-Host "   Start-Process 'http://localhost:5000'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Check running services:" -ForegroundColor White
Write-Host "   docker-compose ps" -ForegroundColor Gray
Write-Host ""
Write-Host "3. View logs:" -ForegroundColor White
Write-Host "   docker-compose logs -f" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Health check:" -ForegroundColor White
Write-Host "   .\scripts\health_check.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan

# Open MLflow
Write-Host ""
$openBrowser = Read-Host "Open MLflow UI in browser? (Y/n)"
if ($openBrowser -ne 'n' -and $openBrowser -ne 'N') {
    Start-Process "http://localhost:5000"
}
