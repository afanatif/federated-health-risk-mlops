#!/bin/bash
set -e

echo ""
echo "========================================================"
echo "     Federated Learning Launcher (with sanity checks)"
echo "========================================================"
echo ""

# Fix PYTHONPATH to root of repo
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to $PYTHONPATH"
echo ""

###############################################################################
# 1. SANITY CHECKS
###############################################################################
echo "Running Sanity Checks..."
echo "--------------------------------------------------------"

# Helper function
check_file() {
  if [ -f "$1" ]; then
    echo "[ OK ] File exists: $1"
  else
    echo "[ERROR] Missing file: $1"
    exit 1
  fi
}

check_dir() {
  if [ -d "$1" ]; then
    echo "[ OK ] Directory exists: $1"
  else
    echo "[ERROR] Missing directory: $1"
    exit 1
  fi
}

# Required directories
check_dir "server"
check_dir "clients"
check_dir "clients/node1"
check_dir "clients/node2"
check_dir "clients/node3"
check_dir "data"
check_dir "models"

# Required files
check_file "data/download.py"
check_file "server/server_flower.py"
check_file "models/model.py"

check_file "clients/node1/client_flower.py"
check_file "clients/node2/client_flower.py"
check_file "clients/node3/client_flower.py"

echo "--------------------------------------------------------"
echo "Sanity checks passed!"
echo ""

###############################################################################
# 2. DATA DOWNLOAD / PREP
###############################################################################
echo "Downloading / verifying dataset..."
python data/download.py
echo "Dataset OK."
echo ""

###############################################################################
# 3. START SERVER
###############################################################################
echo "Starting Flower Server..."
python server/server_flower.py &
SERVER_PID=$!
sleep 3
echo "Server started with PID $SERVER_PID"
echo ""

###############################################################################
# 4. START CLIENTS
###############################################################################
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
echo "Nodes to run: ${NODE_IDS[@]}"

PIDS=()

echo ""
echo "Starting all client nodes..."
echo "--------------------------------------------------------"
for id in "${NODE_IDS[@]}"; do
  CS="clients/node${id}/client_flower.py"
  echo "Launching: $CS"
  python "$CS" &
  PIDS+=($!)
  sleep 1
done

###############################################################################
# 5. WAIT FOR CLIENTS
###############################################################################
echo ""
echo "Waiting for all clients to finish..."
for p in "${PIDS[@]}"; do
  wait "$p" || true
done

###############################################################################
# 6. SHUTDOWN SERVER
###############################################################################
echo "Shutting down server..."
kill $SERVER_PID || true

echo ""
echo "========================================================"
echo " Federated Learning Session Finished Successfully"
echo "========================================================"
echo ""
