# Quick Start Guide

## Prerequisites

**Only requirement:** Docker Desktop

- Windows: https://www.docker.com/products/docker-desktop
- Mac: https://www.docker.com/products/docker-desktop
- Linux: `sudo apt-get install docker-compose`

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/federated-health-risk-mlops.git
cd federated-health-risk-mlops
```

### 2. Start All Services
```bash
docker-compose up -d --build
```

**Wait 30 seconds** for all containers to start.

### 3. Verify Services

```powershell
# Check all containers are running
docker-compose ps

# Should show:
# fl-mlflow       Up      0.0.0.0:5000->5000/tcp
# fl-server       Up      0.0.0.0:8080->8080/tcp
# fl-client-node1 Up
# fl-client-node2 Up
# fl-client-node3 Up
# fl-prometheus   Up      0.0.0.0:9090->9090/tcp
```

### 4. Run Demo

```bash
# Copy demo script
docker cp demos/federated_3_rounds.py fl-mlflow:/tmp/

# Execute
docker exec fl-mlflow python /tmp/federated_3_rounds.py
```

### 5. View Results

Open browser: http://localhost:5000

You'll see:
- Experiment: `yolov8-health-risk-detection`
- Run: `federated_round_1_2_3`
- Metrics with line graphs

## Troubleshooting

### Port Already in Use

**Error:** `Bind for 0.0.0.0:5000 failed: port is already allocated`

**Fix:**
```powershell
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Docker Not Running

**Error:** `Cannot connect to the Docker daemon`

**Fix:** Start Docker Desktop and wait for it to fully initialize

### Containers Exiting

**Fix:**
```bash
# View logs
docker-compose logs server

# Clean rebuild
docker-compose down -v
docker-compose up -d --build
```

### MLflow Not Accessible

```bash
# Check container
docker ps | grep mlflow

# Restart
docker-compose restart mlflow

# Wait 10 seconds, then try http://localhost:5000
```

## Next Steps

1. ✅ MLflow running at http://localhost:5000
2. ✅ Demo completed with metrics
3. 📖 Read [DEPLOYMENT.md](DEPLOYMENT.md) for advanced usage
4. 🚀 Customize `docker-compose.yml` for your needs

## Clean Up

```bash
# Stop all services
docker-compose down

# Remove volumes (deletes data)
docker-compose down -v
```

## Support

If you encounter issues:
1. Check `docker-compose logs`
2. Ensure Docker Desktop is running
3. Try clean rebuild: `docker-compose down -v && docker-compose up -d --build`
4. Open GitHub issue with logs

**Everything should work with just Docker - no Python installation needed!**
