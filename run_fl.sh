#!/bin/bash
set -e

# nodes to run. default 1,2,3
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
echo "Nodes to run: ${NODE_IDS[@]}"

# ensure data
python data/download.py

# start server
python server/server_flower.py &
SERVER_PID=$!
echo "Started server pid=$SERVER_PID"
sleep 3

PIDS=()
for id in "${NODE_IDS[@]}"; do
  cs="clients/node${id}/client_flower.py"
  if [ ! -f "$cs" ]; then
    echo "Client script missing: $cs"
    kill $SERVER_PID || true
    exit 1
  fi
  echo "Starting $cs"
  python "$cs" &
  PIDS+=($!)
  sleep 1
done

# wait for clients
for p in "${PIDS[@]}"; do
  wait "$p" || true
done

kill $SERVER_PID || true
echo "Done"
