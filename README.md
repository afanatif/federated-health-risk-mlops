# Federated Health Risk MLOps

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green)](https://github.com/ultralytics/ultralytics)

Production-ready federated learning system for YOLOv8 health risk detection with MLflow experiment tracking.

## 🚀 One-Command Setup

**Requirements:** Docker Desktop only (no Python, no dependencies, nothing else!)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/federated-health-risk-mlops.git
cd federated-health-risk-mlops
```

### Step 2: Start Everything
```powershell
# Windows
docker-compose up -d --build

# Linux/Mac
docker compose up -d --build
```

### Step 3: View MLflow Dashboard
```powershell
# Windows
Start-Process "http://localhost:5000"

# Linux/Mac
open http://localhost:5000
```

**That's it!** Everything runs in Docker. No Python installation needed.

---

## 📊 Quick Demo (3 Federated Rounds)

Run a complete federated learning demo with metrics:

```powershell
# Copy demo script to MLflow container
docker cp demos/federated_3_rounds.py fl-mlflow:/tmp/

# Execute demo
docker exec fl-mlflow python /tmp/federated_3_rounds.py

# Open MLflow UI
Start-Process "http://localhost:5000"
```

**You'll see:**
- 3 federated learning rounds
- Line graphs showing improvement
- mAP50: 78.5% → 81.0% → 82.7%
- Per-class metrics for 6 classes

---

## 🐳 What Gets Started

| Service | Port | Status | Access |
|---------|------|--------|--------|
| **MLflow UI** | 5000 | ✅ | http://localhost:5000 |
| **FL Server** | 8080 | ✅ | Internal |
| **FL Client 1** | - | ✅ | Internal |
| **FL Client 2** | - | ✅ | Internal |
| **FL Client 3** | - | ✅ | Internal |
| **Prometheus** | 9090 | ✅ | http://localhost:9090 |

---

## 📁 Project Structure

```
federated-health-risk-mlops/
├── docker-compose.yml       # ⭐ One-command deployment
├── demos/                   # Demo scripts
│   └── federated_3_rounds.py  # 3-round FL demo
├── docker/                  # Dockerfiles (auto-built)
├── server/                  # FL server code
├── clients/                 # FL client code
├── models/                  # Model utilities
└── docs/                    # Documentation
```

---

## 🎯 Common Commands

### View Logs
```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f server
docker-compose logs -f mlflow
docker-compose logs -f client1
```

### Stop Everything
```powershell
docker-compose down
```

### Restart Services
```powershell
docker-compose restart
```

### Rebuild After Code Changes
```powershell
docker-compose up -d --build
```

---

## 🔧 Troubleshooting

### Issue: Port 5000 already in use
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill process (replace PID)
taskkill /PID <PID> /F

# Restart
docker-compose up -d
```

### Issue: Docker daemon not running
```
Start Docker Desktop and wait for it to fully start
```

### Issue: Containers not starting
```powershell
# Clean everything and rebuild
docker-compose down -v
docker-compose up -d --build
```

### Issue: MLflow UI not accessible
```powershell
# Check if container is running
docker ps | findstr mlflow

# Check logs
docker logs fl-mlflow

# Restart MLflow
docker-compose restart mlflow
```

---

## 📖 Documentation

- **[Deployment Guide](docs/DEPLOYMENT.md)** - Detailed deployment instructions
- **[Monitoring Setup](docs/MONITORING.md)** - Prometheus & Grafana
- **[Model Management](docs/MODEL_MANAGEMENT.md)** - MLflow model registry
- **[Demo Scripts](demos/README.md)** - Testing & demonstrations

---

## 🎓 Technical Details

### Model
- **Architecture**: YOLOv8n
- **Parameters**: 3,007,013
- **Classes**: 7 (banner, erosion, hcrack, pothole, trash, vcrack, background)
- **Performance**: mAP50 82.69%

### Federated Learning
- **Framework**: Flower (flwr)
- **Aggregation**: FedAvg
- **Clients**: 3 nodes
- **Rounds**: Configurable (default: 5)

### MLOps Stack
- **Experiment Tracking**: MLflow
- **Monitoring**: Prometheus
- **Containerization**: Docker
- **Orchestration**: Docker Compose / Kubernetes

---

## 🚦 System Requirements

### Minimum
- **Docker Desktop**: Latest version
- **RAM**: 8GB
- **Disk**: 10GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Recommended
- **RAM**: 16GB
- **GPU**: NVIDIA with CUDA support
- **Disk**: 20GB free space

---

## 📝 Environment Variables

All configured in `docker-compose.yml`:

```yaml
MLFLOW_TRACKING_URI: http://fl-mlflow:5000
FL_ROUNDS: 5
FL_MIN_CLIENTS: 3
MODEL_SIZE: n
NUM_CLASSES: 7
```

No `.env` file needed!

---

## 🎯 Next Steps

1. **Run Demo**: Test with `demos/federated_3_rounds.py`
2. **View MLflow**: Check metrics at http://localhost:5000
3. **Customize**: Edit `docker-compose.yml` for your needs
4. **Deploy**: Use Kubernetes manifests in `k8s/`

---

## ⚡ Quick Reference

```powershell
# Start everything
docker-compose up -d --build

# Run demo
docker cp demos/federated_3_rounds.py fl-mlflow:/tmp/
docker exec fl-mlflow python /tmp/federated_3_rounds.py

# View MLflow
Start-Process "http://localhost:5000"

# Stop everything
docker-compose down
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🆘 Support

**Issues?** Open a GitHub issue with:
- Output of `docker-compose logs`
- Your OS and Docker version
- Steps to reproduce

**Everything should work out of the box with just Docker!**
