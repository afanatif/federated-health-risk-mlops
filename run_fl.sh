#!/usr/bin/env bash
# run_fl.sh - robust launcher for the federated demo (sanity checks, data prep, server & clients)
# Usage:
#   chmod +x run_fl.sh
#   ./run_fl.sh
set -euo pipefail

# ---------- helpers ----------
log()   { printf "\n[ %s ] %s\n" "$(date +'%H:%M:%S')" "$*"; }
err()   { printf "\n[ %s ] ERROR: %s\n" "$(date +'%H:%M:%S')" "$*" >&2; }
cleanup() {
  log "Cleaning up processes..."
  # kill clients if any
  if [ "${CLIENT_PIDS+x}" = "x" ] && [ ${#CLIENT_PIDS[@]:-0} -gt 0 ]; then
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

# ---------- find repo root ----------
# prefer git top-level if available
GIT_TOPLEVEL=""
if command -v git >/dev/null 2>&1; then
  GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null || true)
fi

if [ -n "$GIT_TOPLEVEL" ]; then
  REPO_ROOT="$GIT_TOPLEVEL"
else
  # fallback: search upward for a directory that contains server/server_flower.py
  CWD="$(pwd)"
  FOUND=""
  SEARCH_DIR="$CWD"
  for i in 0 1 2 3 4; do
    if [ -f "$SEARCH_DIR/server/server_flower.py" ]; then
      FOUND="$SEARCH_DIR"
      break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
  done
  if [ -n "$FOUND" ]; then
    REPO_ROOT="$FOUND"
  else
    # last fallback: search downward (project may be nested)
    FOUND_DOWN=$(find . -maxdepth 3 -type f -path "*/server/server_flower.py" -print -quit || true)
    if [ -n "$FOUND_DOWN" ]; then
      REPO_ROOT="$(cd "$(dirname "$(dirname "$FOUND_DOWN")")" && pwd)"
    else
      # as final fallback use current dir
      REPO_ROOT="$(pwd)"
    fi
  fi
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
log "Working directory set to: $REPO_ROOT"
log "PYTHONPATH = $PYTHONPATH"

# ---------- locate sanity script ----------
# prefer scripts/sanity_check.py directly in repo root
SANITY_SCRIPT="$REPO_ROOT/scripts/sanity_check.py"
if [ ! -f "$SANITY_SCRIPT" ]; then
  # try to find any sanity_check.py within 3 levels
  SANITY_SCRIPT=$(find "$REPO_ROOT" -maxdepth 3 -type f -name "sanity_check.py" -print -quit || true)
fi

if [ -z "$SANITY_SCRIPT" ] || [ ! -f "$SANITY_SCRIPT" ]; then
  err "Sanity check script not found. Searched for scripts/sanity_check.py and fallback locations."
  err "Please ensure scripts/sanity_check.py exists in the repo (path reported: $SANITY_SCRIPT)"
  exit 1
fi

log "Running sanity check at: $SANITY_SCRIPT"
if python "$SANITY_SCRIPT"; then
  log "Sanity check PASSED."
else
  err "Sanity check FAILED. Fix reported issues before running."
  exit 1
fi

# ---------- verify baseline files & folders ----------
REQUIRED_DIRS=( "server" "clients" "data" )
for d in "${REQUIRED_DIRS[@]}"; do
  if [ ! -d "$REPO_ROOT/$d" ]; then
    err "Required directory missing: $REPO_ROOT/$d"
    exit 1
  fi
done

if [ ! -f "$REPO_ROOT/data/download.py" ]; then
  err "Required file missing: data/download.py"
  exit 1
fi

# accept model located at models/model.py or model.py
if [ -f "$REPO_ROOT/models/model.py" ]; then
  MODEL_PATH="$REPO_ROOT/models/model.py"
elif [ -f "$REPO_ROOT/model.py" ]; then
  MODEL_PATH="$REPO_ROOT/model.py"
else
  err "Model file not found at models/model.py or model.py"
  exit 1
fi
log "Model file located: $MODEL_PATH"

# ---------- data download (skip if present) ----------
DATA_PRESENT=false
if [ -d "$REPO_ROOT/clients/node1/data/images" ] && [ "$(ls -A "$REPO_ROOT/clients/node1/data/images" 2>/dev/null | wc -l)" -gt 0 ]; then
  DATA_PRESENT=true
fi

if [ "$DATA_PRESENT" = true ]; then
  log "Data appears already present; skipping data download."
else
  log "Data not present. Running data/download.py to fetch and extract node archives..."
  python "$REPO_ROOT/data/download.py"
  log "Data download/extract completed."
fi

# ---------- start server ----------
log "Starting Flower server (server/server_flower.py)..."
python "$REPO_ROOT/server/server_flower.py" &
SERVER_PID=$!
log "Server started with PID $SERVER_PID"

# ---------- wait for server port ----------
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

# ---------- start clients ----------
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
log "Nodes configured to run: ${NODE_IDS[*]}"

CLIENT_PIDS=()
for id in "${NODE_IDS[@]}"; do
  CLIENT_SCRIPT="$REPO_ROOT/clients/node${id}/client_flower.py"
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
  sleep 0.5
done

# ---------- wait for clients ----------
log "Waiting for clients (PIDs: ${CLIENT_PIDS[*]}) to finish. Server PID: $SERVER_PID"
for p in "${CLIENT_PIDS[@]}"; do
  if kill -0 "$p" 2>/dev/null; then
    wait "$p" || true
    log "Client pid $p finished"
  else
    log "Client pid $p not running (may have exited early)"
  fi
done

# ---------- shutdown server ----------
if kill -0 "$SERVER_PID" 2>/dev/null; then
  log "All clients finished. Terminating server pid $SERVER_PID"
  kill "$SERVER_PID" || true
  sleep 1
else
  log "Server process already exited."
fi

log "Federated run completed successfully."
exit 0
