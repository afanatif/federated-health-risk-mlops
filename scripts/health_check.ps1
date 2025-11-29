# Health Check Script for Windows
# Verifies all services are running correctly

Write-Host "🔍 Checking Federated Learning System Health..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check Docker
Write-Host ""
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host "   Please start Docker Desktop" -ForegroundColor Yellow
    exit 1
}

# Check containers
Write-Host ""
Write-Host "📦 Container Status:" -ForegroundColor Yellow
docker-compose ps

# Check MLflow
Write-Host ""
Write-Host "🔍 Checking MLflow (port 5000)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 5 -UseBasicParsing
    Write-Host "✅ MLflow is accessible" -ForegroundColor Green
}
catch {
    Write-Host "❌ MLflow is not accessible" -ForegroundColor Red
    Write-Host "   Try: docker-compose restart mlflow" -ForegroundColor Yellow
}

# Check Server
Write-Host ""
Write-Host "🔍 Checking FL Server (port 8080)..." -ForegroundColor Yellow
$serverTest = Test-NetConnection -ComputerName localhost -Port 8080 -WarningAction SilentlyContinue
if ($serverTest.TcpTestSucceeded) {
    Write-Host "✅ FL Server is listening" -ForegroundColor Green
}
else {
    Write-Host "⚠️  FL Server port not accessible (may be normal if no training)" -ForegroundColor Yellow
}

# Check Prometheus
Write-Host ""
Write-Host "🔍 Checking Prometheus (port 9090)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090" -TimeoutSec 5 -UseBasicParsing
    Write-Host "✅ Prometheus is accessible" -ForegroundColor Green
}
catch {
    Write-Host "⚠️  Prometheus is not accessible" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "🎯 Quick Actions:" -ForegroundColor Cyan
Write-Host "   View logs: docker-compose logs -f"
Write-Host "   Restart:   docker-compose restart"
Write-Host "   MLflow UI: http://localhost:5000"
Write-Host "================================================" -ForegroundColor Cyan
