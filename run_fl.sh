#!/bin/bash
export PYTHONPATH=$(pwd)

set -e

# nodes to run. default 1,2,3
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
echo "Nodes to run: ${NODE_IDS[@]}"

# -----------------------
# Ensure data
# -----------------------
python data/download.py

# -----------------------
# Start server
# -----------------------
python server/server_flower.py &
SERVER_PID=$!
echo "Started server pid=$SERVER_PID"

# Wait for server to open port 8080
echo "Waiting for server to be ready..."
while ! nc -z localhost 8080; do
  sleep 1
done
echo "Server ready!"

# -----------------------
# Start clients
# -----------------------
PIDS=()
for id in "${NODE_IDS[@]}"; do
  CLIENT_SCRIPT="clients/node${id}/client_flower.py"
  if [ ! -f "$CLIENT_SCRIPT" ]; then
    echo "Client script missing: $CLIENT_SCRIPT"
    kill $SERVER_PID || true
    exit 1
  fi
  echo "Starting $CLIENT_SCRIPT"
  python "$CLIENT_SCRIPT" &
  PIDS+=($!)
  sleep 1
done

# -----------------------
# Wait for clients to finish
# -----------------------
for p in "${PIDS[@]}"; do
  wait "$p" || true
done

# -----------------------
# Stop server
# -----------------------
kill $SERVER_PID || true
echo "All nodes finished. Server stopped."
