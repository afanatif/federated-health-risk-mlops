#!/bin/bash
set -e

cd "$(dirname "$0")/.." || true
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to $PYTHONPATH"

: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"

# sanity: ensure data downloader exists
if [ ! -f data/download.py ]; then
  echo "Missing data/download.py"
  exit 1
fi

# download datasets
python data/download.py

# start server
python server/server_flower.py &
SERVER_PID=$!
echo "Server started with pid $SERVER_PID"

# wait for port 8080 to be open, fail if server dies first
MAX_WAIT=20
COUNT=0
while ! nc -z localhost 8080; do
  sleep 0.5
  COUNT=$((COUNT + 1))
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server process died. Check server logs. Exiting."
    wait $SERVER_PID || true
    exit 1
  fi
  if [ $COUNT -gt $MAX_WAIT ]; then
    echo "Timeout waiting for server port 8080. Server may not be ready. Exiting."
    kill $SERVER_PID || true
    exit 1
  fi
done
echo "Server is listening on 8080."

# start clients
PIDS=()
for id in "${NODE_IDS[@]}"; do
  cs="clients/node${id}/client_flower.py"
  if [ ! -f "$cs" ]; then
    echo "Missing client script: $cs"
    kill $SERVER_PID || true
    exit 1
  fi
  echo "Starting $cs"
  python "$cs" &
  PIDS+=($!)
  sleep 0.5
done

for p in "${PIDS[@]}"; do
  wait "$p" || true
done

kill $SERVER_PID || true
echo "Done"
