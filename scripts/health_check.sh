#!/bin/bash
# Health check script for federated learning deployment

echo "🔍 Checking Federated Learning System Health..."
echo "================================================"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "   Please start Docker Desktop"
    exit 1
fi
echo "✅ Docker is running"

# Check containers
echo ""
echo "📦 Container Status:"
docker-compose ps

# Check MLflow
echo ""
echo "🔍 Checking MLflow (port 5000)..."
if curl -s http://localhost:5000 > /dev/null; then
    echo "✅ MLflow is accessible"
else
    echo "❌ MLflow is not accessible"
    echo "   Try: docker-compose restart mlflow"
fi

# Check Server
echo ""
echo "🔍 Checking FL Server (port 8080)..."
if nc -z localhost 8080 2>/dev/null || (echo > /dev/tcp/localhost/8080) 2>/dev/null; then
    echo "✅ FL Server is listening"
else
    echo "⚠️  FL Server port not accessible (may be normal if no training)"
fi

# Check Prometheus
echo ""
echo "🔍 Checking Prometheus (port 9090)..."
if curl -s http://localhost:9090 > /dev/null; then
    echo "✅ Prometheus is accessible"
else
    echo "⚠️  Prometheus is not accessible"
fi

echo ""
echo "================================================"
echo "🎯 Quick Actions:"
echo "   View logs: docker-compose logs -f"
echo "   Restart:   docker-compose restart"
echo "   MLflow UI: http://localhost:5000"
echo "================================================"
