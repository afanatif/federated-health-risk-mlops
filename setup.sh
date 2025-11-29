#!/bin/bash
# Setup Verification Script for Linux/Mac
# Run this after cloning from GitHub to ensure everything works

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================================"
echo -e "  Federated Learning MLOps - Setup Verification"
echo -e "======================================================================${NC}"
echo ""

# Step 1: Check Docker
echo -e "${YELLOW}[1/5] Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo ""
    echo -e "${YELLOW}Please install Docker:${NC}"
    echo "  Mac: https://www.docker.com/products/docker-desktop"
    echo "  Linux: sudo apt-get install docker-compose"
    exit 1
fi

DOCKER_VERSION=$(docker --version)
echo -e "${GREEN}✅ Docker installed: $DOCKER_VERSION${NC}"

if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker daemon is not running!${NC}"
    echo -e "${YELLOW}Please start Docker Desktop${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker daemon is running${NC}"

# Step 2: Check required files
echo ""
echo -e "${YELLOW}[2/5] Checking required files...${NC}"
REQUIRED_FILES=(
    "docker-compose.yml"
    "docker/Dockerfile.server"
    "docker/Dockerfile.client"
    "docker/Dockerfile.mlflow"
    "demos/federated_3_rounds.py"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found: $file${NC}"
    else
        echo -e "${RED}❌ Missing: $file${NC}"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    echo ""
    echo -e "${RED}❌ Some required files are missing!${NC}"
    echo -e "${YELLOW}   Make sure you cloned the complete repository${NC}"
    exit 1
fi

# Step 3: Build and start services
echo ""
echo -e "${YELLOW}[3/5] Building and starting services...${NC}"
echo -e "   This may take 5-10 minutes on first run..."

docker-compose up -d --build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start services!${NC}"
    echo -e "${YELLOW}   Check logs with: docker-compose logs${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Services started${NC}"

# Step 4: Wait for services to be ready
echo ""
echo -e "${YELLOW}[4/5] Waiting for services to be ready...${NC}"
echo "   Waiting 30 seconds..."
sleep 30

# Check MLflow
MLFLOW_READY=false
for i in {1..10}; do
    if curl -s http://localhost:5000 > /dev/null; then
        MLFLOW_READY=true
        break
    fi
    echo "   Attempt $i/10: MLflow not ready yet..."
    sleep 3
done

if [ "$MLFLOW_READY" = true ]; then
    echo -e "${GREEN}✅ MLflow is accessible at http://localhost:5000${NC}"
else
    echo -e "${YELLOW}⚠️  MLflow may not be ready yet${NC}"
    echo "   Check with: docker logs fl-mlflow"
fi

# Step 5: Run demo
echo ""
echo -e "${YELLOW}[5/5] Running demo (3 federated rounds)...${NC}"

docker cp demos/federated_3_rounds.py fl-mlflow:/tmp/
docker exec fl-mlflow python /tmp/federated_3_rounds.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Demo completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Demo execution had issues${NC}"
    echo "   You can run it manually later"
fi

# Summary
echo ""
echo -e "${CYAN}======================================================================"
echo -e "  Setup Complete!"
echo -e "======================================================================${NC}"
echo ""
echo -e "${CYAN}🎯 Next Steps:${NC}"
echo ""
echo -e "${NC}1. View MLflow Dashboard:${NC}"
echo "   open http://localhost:5000  (Mac)"
echo "   xdg-open http://localhost:5000  (Linux)"
echo ""
echo "2. Check running services:"
echo "   docker-compose ps"
echo ""
echo "3. View logs:"
echo "   docker-compose logs -f"
echo ""
echo "4. Health check:"
echo "   ./scripts/health_check.sh"
echo ""
echo -e "${CYAN}======================================================================${NC}"
