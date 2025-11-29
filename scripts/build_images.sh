#!/usr/bin/env bash
# Build Docker images for federated learning system
# Usage: ./scripts/build_images.sh [--push]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-federated-learning}"
PUSH_IMAGES=false

# Parse arguments
if [[ "$1" == "--push" ]]; then
    PUSH_IMAGES=true
    echo "Will push images to registry after building"
fi

echo "============================================"
echo "Building Docker Images for Federated Learning"
echo "============================================"
echo "Version: $VERSION"
echo "Registry: $REGISTRY"
echo ""

# Build server image
echo "📦 Building server image..."
docker build -f docker/Dockerfile.server -t "${REGISTRY}/server:${VERSION}" .
echo "✅ Server image built: ${REGISTRY}/server:${VERSION}"
echo ""

# Build client image
echo "📦 Building client image..."
docker build -f docker/Dockerfile.client -t "${REGISTRY}/client:${VERSION}" .
echo "✅ Client image built: ${REGISTRY}/client:${VERSION}"
echo ""

# Build MLflow image
echo "📦 Building MLflow image..."
docker build -f docker/Dockerfile.mlflow -t "${REGISTRY}/mlflow:${VERSION}" .
echo "✅ MLflow image built: ${REGISTRY}/mlflow:${VERSION}"
echo ""

# Tag as latest
echo "📌 Tagging images as latest..."
docker tag "${REGISTRY}/server:${VERSION}" "${REGISTRY}/server:latest"
docker tag "${REGISTRY}/client:${VERSION}" "${REGISTRY}/client:latest"
docker tag "${REGISTRY}/mlflow:${VERSION}" "${REGISTRY}/mlflow:latest"
echo "✅ Images tagged as latest"
echo ""

# Push images if requested
if $PUSH_IMAGES; then
    echo "🚀 Pushing images to registry..."
    docker push "${REGISTRY}/server:${VERSION}"
    docker push "${REGISTRY}/server:latest"
    docker push "${REGISTRY}/client:${VERSION}"
    docker push "${REGISTRY}/client:latest"
    docker push "${REGISTRY}/mlflow:${VERSION}"
    docker push "${REGISTRY}/mlflow:latest"
    echo "✅ Images pushed successfully"
    echo ""
fi

echo "============================================"
echo "Build Complete!"
echo "============================================"
echo "Images built:"
echo "  - ${REGISTRY}/server:${VERSION}"
echo "  - ${REGISTRY}/client:${VERSION}"
echo "  - ${REGISTRY}/mlflow:${VERSION}"
echo ""
echo "To run locally with Docker Compose:"
echo "  docker-compose up -d"
echo ""
echo "To deploy to Kubernetes:"
echo "  ./scripts/deploy_k8s.sh"
echo "============================================"
