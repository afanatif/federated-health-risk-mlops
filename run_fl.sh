#!/bin/bash
set -e

echo "=== Federated Learning Pipeline Starting ==="

# --------------------------------------------------------
# 0. PYTHONPATH
# --------------------------------------------------------
export PYTHONPATH=$(pwd)
echo "PYTHONPATH = $PYTHONPATH"

# --------------------------------------------------------
# 1. RUN SANITY CHECK FIRST
# --------------------------------------------------------
if [ ! -f "sanity_check.py" ]; then
  echo "❌ sanity_check.py not found!"
  exit 1
fi

echo "=== Running sanity_check.py ==="
python sanity_check.py || { echo "❌ Sanity check failed!"; exit 1; }
echo "✓ Sanity check passed"

# --------------------------------------------------------
# 2. NODES TO RUN
# --------------------------------------------------------
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
echo "Nodes to run: ${NODE_IDS[@]}"

# --------------------------------------------------------
# 3. DATA DOWNLOAD (skip if exists)
# --------------------------------------------------------
if [ -d "data" ]; then
  echo "✓ Data folder exists, skipping download"
else
  echo "=== Downloading dataset ==="
  if [ ! -f "data/download.py" ]; then
      echo "❌ Missing: data/download.py"
      exit 1
  fi
  python data/download.py
fi

# --------------------------------------------------------
# 4. VERIFY MODEL FILE EXISTS
# --------------------------------------------------------
if [ ! -f "model/model.py" ]; then
  echo "❌ model/model.py missing!"
  exit 1
else
  echo "✓ Found model/model.py"
fi

# --------------------------------------------------------
# 5. START FEDERATED SERVER
# --------------------------------------------------------
if [ ! -f "server/server_flower.py" ]; then
  echo "❌ server_flower.py missing!"
  exit 1
fi

echo "=== Starting Flower Server ==="
python server/server_flower.py &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"
sleep 3

# --------------------------------------------------------
# 6. START CLIENTS
# --------------------------------------------------------
PIDS=()

for id in "${NODE_IDS[@]}"; do
  CLIENT_PATH="clients/node${id}/client_flower.py"

  if [ ! -f "$CLIENT_PATH" ]; then
      echo "❌ Client missing: $CLIENT_PATH"
      kill $SERVER_PID || true
      exit 1
  fi

  echo "=== Starting Client for Node $id ==="
  python "$CLIENT_PATH" &
  PIDS+=($!)
  sleep 1
done

# --------------------------------------------------------
# 7. WAIT FOR CLIENTS TO FINISH
# --------------------------------------------------------
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

# --------------------------------------------------------
# 8. SHUTDOWN SERVER
# --------------------------------------------------------
echo "Shutting down server..."
kill $SERVER_PID || true

echo "=== Federated Learning Run Completed ==="
