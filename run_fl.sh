#!/bin/bash
set -e
# ensure script runs relative to repo root
cd "$(dirname "$0")/.." || true

export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to $PYTHONPATH"

# nodes to run. default 1,2,3
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
echo "Nodes to run: ${NODE_IDS[@]}"

# Ensure data (your script already handles downloads/unzip)
python data/download.py

# Start server in background
echo "Starting Flower server..."
python server/server_flower.py &
SERVER_PID=$!
echo "Started server pid=$SERVER_PID"

# Wait for server to be ready (listen on 8080)
echo "Waiting for server to open port 8080..."
# use netcat if available, otherwise simple sleep fallback
if command -v nc >/dev/null 2>&1; then
  until nc -z localhost 8080; do
    sleep 0.5
  done
else
  # fallback: wait a few seconds (not ideal but keep demo simple)
  sleep 5
fi
echo "Server is reachable."

# Start clients
PIDS=()
for id in "${NODE_IDS[@]}"; do
  CLIENT_SCRIPT="clients/node${id}/client_flower.py"
  if [ ! -f "$CLIENT_SCRIPT" ]; then
    echo "Client script missing: $CLIENT_SCRIPT"
    kill $SERVER_PID || true
    exit 1
  fi
  echo "Starting $CLIENT_SCRIPT ..."
  python "$CLIENT_SCRIPT" &
  PIDS+=($!)
  # small stagger so clients register cleanly
  sleep 0.5
done

# Wait for clients to complete
for p in "${PIDS[@]}"; do
  wait "$p" || true
done

# Stop server
kill $SERVER_PID || true
echo "All clients finished; server stopped."
