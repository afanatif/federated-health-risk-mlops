#!/usr/bin/env bash
# run_fl.sh - robust launcher for the federated demo (sanity checks, data prep, server & clients)
# Usage:
#   make executable: chmod +x run_fl.sh
#   run: ./run_fl.sh
# Optional: export NODES_TO_RUN="1,2,3" to control which nodes start

set -euo pipefail

# ---------- Helper functions ----------
log()   { printf "\n[ %s ] %s\n" "$(date +'%H:%M:%S')" "$*"; }
err()   { printf "\n[ %s ] ERROR: %s\n" "$(date +'%H:%M:%S')" "$*" >&2; }
cleanup() {
  log "Cleaning up processes..."
  # kill clients if any
  if [ "${CLIENT_PIDS+x}" = "x" ] && [ ${#CLIENT_PIDS[@]} -gt 0 ]; then
    for p in "${CLIENT_PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        log "Killing client pid $p"
        kill "$p" || true
      fi
    done
  fi
  # kill server if running
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Killing server pid $SERVER_PID"
    kill "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

# ---------- Step 0: ensure script runs from repo root ----------
# Move to repo root (script expected in repo root). If run_fl.sh is inside a folder,
# this will attempt to move to parent folder of script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." 2>/dev/null || true
REPO_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT"
log "Working directory set to: $REPO_ROOT"
log "PYTHONPATH = $PYTHONPATH"

# ---------- Step 1: Sanity check ----------
SANITY_SCRIPT="scripts/sanity_check.py"
if [ ! -f "$SANITY_SCRIPT" ]; then
  err "Sanity check script not found at $SANITY_SCRIPT"
  err "Make sure scripts/sanity_check.py exists (you previously created scripts/sanity_check.py)."
  exit 1
fi
log "Running sanity check: $SANITY_SCRIPT"
if python "$SANITY_SCRIPT"; then
  log "Sanity check passed."
else
  err "Sanity check FAILED. Fix reported issues before running."
  exit 1
fi

# ---------- Step 2: Basic file/directory checks ----------
# Required paths (tweak if your layout differs)
REQUIRED_DIRS=( "server" "clients" "data" "clients/node1" "clients/node2" "clients/node3" )
for d in "${REQUIRED_DIRS[@]}"; do
  if [ ! -d "$d" ]; then
    err "Required directory missing: $d"
    exit 1
  fi
done
REQUIRED_FILES=( "server/server_flower.py" "data/download.py" )
for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    err "Required file missing: $f"
    exit 1
  fi
done
# model file can exist as models/model.py or model.py; accept both
if [ -f "models/model.py" ]; then
  MODEL_PATH="models/model.py"
elif [ -f "model.py" ]; then
  MODEL_PATH="model.py"
else
  err "Model file not found at models/model.py or model.py"
  exit 1
fi
log "Model file located: $MODEL_PATH"

# ---------- Step 3: Prepare / download data (skip if already present) ----------
# We consider data present if clients/node1/data/images exists and is non-empty
DATA_PRESENT=false
if [ -d "clients/node1/data/images" ] && [ "$(ls -A clients/node1/data/images 2>/dev/null | wc -l)" -gt 0 ]; then
  DATA_PRESENT=true
fi

if [ "$DATA_PRESENT" = true ]; then
  log "Data appears already present; skipping data download."
else
  log "Data not present. Running data/download.py to fetch and extract node archives..."
  python data/download.py
  log "Data download/extract completed."
fi

# ---------- Step 4: Start server ----------
log "Starting Flower server (server/server_flower.py)..."
python server/server_flower.py &
SERVER_PID=$!
log "Server started with PID $SERVER_PID"

# ---------- Step 5: Wait for server port (8080) to be listening ----------
SERVER_ADDR="127.0.0.1"
SERVER_PORT=8080
MAX_WAIT_SEC=30
SLEEP_INTERVAL=0.5
elapsed=0

check_port_with_nc() {
  nc -z "$SERVER_ADDR" "$SERVER_PORT" >/dev/null 2>&1
}

check_port_with_python() {
  python - <<PY - "$SERVER_ADDR" "$SERVER_PORT"
import socket, sys
addr=sys.argv[1]; port=int(sys.argv[2])
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect((addr, port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

log "Waiting for server to start listening on ${SERVER_ADDR}:${SERVER_PORT} (timeout ${MAX_WAIT_SEC}s)..."
if command -v nc >/dev/null 2>&1; then
  while ! check_port_with_nc; do
    sleep "$SLEEP_INTERVAL"
    elapsed=$(awk "BEGIN {print $elapsed+$SLEEP_INTERVAL}")
    # verify server still alive
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      err "Server process $SERVER_PID died. Check server/server_flower.py logs. Exiting."
      exit 1
    fi
    if (( $(echo "$elapsed >= $MAX_WAIT_SEC" | bc -l) )); then
      err "Timeout waiting for server port. Exiting."
      exit 1
    fi
  done
else
  # fallback to Python-based check
  while ! check_port_with_python; do
    sleep "$SLEEP_INTERVAL"
    elapsed=$(awk "BEGIN {print $elapsed+$SLEEP_INTERVAL}")
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      err "Server process $SERVER_PID died. Check server/server_flower.py logs. Exiting."
      exit 1
    fi
    if (( $(echo "$elapsed >= $MAX_WAIT_SEC" | bc -l) )); then
      err "Timeout waiting for server port (python fallback). Exiting."
      exit 1
    fi
  done
fi
log "Server is listening on ${SERVER_ADDR}:${SERVER_PORT}"

# ---------- Step 6: Start clients ----------
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
log "Nodes configured to run: ${NODE_IDS[*]}"

CLIENT_PIDS=()
for id in "${NODE_IDS[@]}"; do
  CLIENT_SCRIPT="clients/node${id}/client_flower.py"
  if [ ! -f "$CLIENT_SCRIPT" ]; then
    err "Client script missing: $CLIENT_SCRIPT"
    log "Shutting down server ($SERVER_PID) and exiting."
    kill "$SERVER_PID" || true
    exit 1
  fi
  log "Starting client node$id -> $CLIENT_SCRIPT"
  python "$CLIENT_SCRIPT" &
  pid=$!
  CLIENT_PIDS+=("$pid")
  log "Client node$id started with pid $pid"
  # small stagger
  sleep 0.5
done

# ---------- Step 7: Monitor clients and server ----------
log "Waiting for clients (PIDs: ${CLIENT_PIDS[*]}) to finish. Server PID: $SERVER_PID"
for p in "${CLIENT_PIDS[@]}"; do
  if kill -0 "$p" 2>/dev/null; then
    wait "$p" || true
    log "Client pid $p finished"
  else
    log "Client pid $p not running (may have exited early)"
  fi
done

# If server still running, stop it gracefully
if kill -0 "$SERVER_PID" 2>/dev/null; then
  log "All clients finished. Terminating server pid $SERVER_PID"
  kill "$SERVER_PID" || true
  # give it a moment
  sleep 1
else
  log "Server process already exited."
fi

log "Federated run completed successfully."

# explicit success exit (trap cleanup will run)
exit 0
