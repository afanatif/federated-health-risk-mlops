# PowerShell script to run Federated Learning Server and Clients
# Uses the specific Python 3.12 executable where dependencies are installed

$python = "C:\Users\aatif\AppData\Local\Programs\Python\Python312\python.exe"
$root = $PSScriptRoot

Write-Host "Starting Federated Learning System..." -ForegroundColor Cyan
Write-Host "Using Python: $python" -ForegroundColor Gray

# 1. Start Server
Write-Host "Launching Server..." -ForegroundColor Green
Start-Process -FilePath $python -ArgumentList "server/server_flower.py" -WorkingDirectory $root -WindowStyle Normal
# Wait for server to initialize
Start-Sleep -Seconds 5

# 2. Start Client 1
Write-Host "Launching Client 1..." -ForegroundColor Yellow
Start-Process -FilePath $python -ArgumentList "clients/node1/client_flower.py" -WorkingDirectory $root -WindowStyle Normal

# 3. Start Client 2
Write-Host "Launching Client 2..." -ForegroundColor Yellow
Start-Process -FilePath $python -ArgumentList "clients/node2/client_flower.py" -WorkingDirectory $root -WindowStyle Normal

# 4. Start Client 3
Write-Host "Launching Client 3..." -ForegroundColor Yellow
Start-Process -FilePath $python -ArgumentList "clients/node3/client_flower.py" -WorkingDirectory $root -WindowStyle Normal

Write-Host "All processes launched in separate windows." -ForegroundColor Cyan
